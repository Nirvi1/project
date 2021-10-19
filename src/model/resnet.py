# -*- coding:utf-8 -*-
"""
File: resnet.py
File Created: 2021-09-23
Author: Nirvi Badyal
"""
import torch
import torch.nn as nn
from torchvision import models

from .helper import LIST_ATTR, grad_set, RoIPoolNetwork

import timm
class ResNet18Model(nn.Module):
    def __init__(self):
        super(ResNet18Model, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #backbone = models.resnet50(pretrained=True).to(device)
        #backbone = models.resnet18(pretrained=True)
        #backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        backbone = timm.create_model('seresnext26d_32x4d', pretrained=True)
        #backbone = timm.create_model('seresnext101_32x8d', pretrained=True)
        #backbone = models.resnext50_32x4d(pretrained=True).to(device)
        backbone = list(backbone.children())
        self.features = nn.Sequential(*backbone[:-2])
        #grad_set(self.features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.downsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        fc_in = backbone[-1].in_features
        for i in range(6):
            fc_attr = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(
                    in_features=fc_in, out_features=LIST_ATTR[i]))
            setattr(self, 'fc' + str(i), fc_attr)

    def forward(self, x):
        x = self.features(x)
        #x = self.downsample(x)
        #print(" upsampled ")
        x = self.avgpool(x)
       # x = self.maxpool(x)
        x = torch.flatten(x, 1)

        output = []
        for i in range(6):
          fc_attr = getattr(self, 'fc' + str(i))
          output.append(fc_attr(x))
        return output

class ResNet101RoI(nn.Module):
    def __init__(self):
        super(ResNet101RoI, self).__init__()

        #backbone = models.resnext50_32x4d(pretrained=True)
        backbone = timm.create_model('seresnext26d_32x4d', pretrained=True)
        backbone = list(backbone.children())
        self.features = nn.Sequential(*backbone[:3])
        #grad_set(self.features)
        self.upsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        # Conv0 output: [B, 64, 112, 112]
        self.glayer = nn.Sequential(*backbone[3:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # ResBlock4 output: [B, 512, 1, 1]

        self.n_ch = 64
        self.roi_in = self.n_ch*8*4
        self.roi_out = 64
        self.roi_fc = nn.Sequential(
            nn.Linear(in_features=112, out_features=112, bias=False),
            nn.BatchNorm2d(num_features=self.n_ch))
        self.roi_pool = RoIPoolNetwork(
                output=[2, 2],
                bbox=[7, 7],
                scale=112./224)
        # [B, 64, 8, 4]
        self.llayer = nn.Sequential(
            nn.Linear(in_features=self.roi_in, 
                      out_features=self.roi_in * 2),
            nn.BatchNorm1d(num_features=self.roi_in * 2),
            # [B, self.roi_in * 2]
            nn.ReLU(),
            nn.Unflatten(1, torch.Size([64, 8, 8])),
            nn.AdaptiveAvgPool2d((1, 1)))
        # llayer output: [B, roi_out]
        fc_in = backbone[-1].in_features + self.roi_out
        self.fusion = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=fc_in,
                      out_features=fc_in * 2),
            nn.ReLU())


        for i in range(6):
            fc_attr = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features=fc_in*2, 
                          out_features=LIST_ATTR[i]))
            setattr(self, 'fc' + str(i), fc_attr)

    def forward(self, x, lm):
        x = self.features(x)
        # print(x.size())

        gx = self.glayer(x)
        #gx = self.upsample(gx)
        gx = self.avgpool(gx)
        gx = torch.flatten(gx, 1)

        lx = self.roi_fc(x)
        lx = self.roi_pool(lx, lm)
        lx = torch.flatten(lx, 1)
        lx = self.llayer(lx)
        lx = torch.flatten(lx, 1)

        fx = torch.cat((gx, lx), dim=1)
        fx = self.fusion(fx)

        output = []
        for i in range(6):
          fc_attr = getattr(self, 'fc' + str(i))
          output.append(fc_attr(fx))
        return output


