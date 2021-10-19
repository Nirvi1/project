import torch.nn.functional as F

# -*- coding:utf-8 -*-
"""
File: resnet.py
File Created: 2021-09-23
Author: nyLiao
"""
import torch
import torch.nn as nn
from torchvision import models

from .helper import LIST_ATTR, grad_set, RoIPoolNetwork

import timm


class ResNet18Model(nn.Module):
    def __init__(self):
        super(ResNet18M6, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # model_arch = models.resnet50(pretrained=True).to(device)
        # model_arch = models.resnet18(pretrained=True)
        # model_arch = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        model_arch = timm.create_model('seresnext101_32x8d', pretrained=True)
        # model_arch = timm.create_model('seresnext101_32x8d', pretrained=True)
        # model_arch = models.resnext50_32x4d(pretrained=True).to(device)
        model_arch = list(model_arch.children())
        self.features = nn.Sequential(*model_arch[:-2])
        # grad_set(self.features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.downsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        fc_in = model_arch[-1].in_features
        for i in range(6):
            fc_attr = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(
                    in_features=fc_in, out_features=ATTR_OUT[i]))
            setattr(self, 'fc' + str(i), fc_attr)

    def forward(self, x):
        x = self.features(x)
        # x = self.downsample(x)
        # print(" upsampled ")
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

        model_arch = models.resnext101_32x8d(pretrained=True)
        # model_arch = timm.create_model('seresnext26d_32x4d', pretrained=True)
        model_arch.eval()
        # model_arch = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        # model_arch = timm.create_model('tf_efficientnet_b0_ap', pretrained=True)
        # model_arch = models.googlenet(pretrained=True)
        model_arch = list(model_arch.children())
        self.features = nn.Sequential(*model_arch[:3])
        grad_set(self.features)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.upsample = F.interpolate(img, scale_factor=(0.5,0.5), mode="bilinear")
        # Conv0 output: [B, 64, 112, 112]
        self.layer_next = nn.Sequential(*model_arch[3:-2])
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.MaxPool2d((1, 1))
        # ResBlock4 output: [B, 512, 1, 1]

        self.n_ch = 64
        self.roi_in = self.n_ch * 8 * 4
        self.roi_out = 64
        self.roi_fc = nn.Sequential(
            nn.Linear(in_features=112, out_features=112, bias=False),
            nn.BatchNorm2d(num_features=self.n_ch))
        self.roi_pooling_layer = RoIPoolNetwork(
                output=[2, 2],
                bbox=[5, 5],
                scale=112./224)
        # [B, 64, 8, 4]
        self.llayer = nn.Sequential(
            nn.Linear(in_features=self.roi_in,
                      out_features=self.roi_in * 2),
            nn.BatchNorm1d(num_features=self.roi_in * 2),
            # [B, self.roi_in * 2]
            nn.ReLU(),
            #  nn.Sigmoid(),
            nn.Unflatten(1, torch.Size([64, 8, 8])),
            nn.AdaptiveAvgPool2d((1, 1)), nn.MaxPool2d((1, 1)))
        # llayer output: [B, roi_out]
        self.uprelu = nn.ReLU()
        self.down = nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0)
        self.up = nn.ConvTranspose1d(16, 32, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm1d(32)
        fc_in = model_arch[-1].in_features + self.roi_out
        self.fusion = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=fc_in,
                      out_features=fc_in * 2),
            nn.ReLU())
        # nn.Sigmoid())

        for i in range(6):
            fc_attr = nn.Sequential(
                nn.Dropout(),
                # nn.Identity(),
                nn.Linear(in_features=fc_in * 2,
                          out_features=LIST_ATTR[i]))
            setattr(self, 'fc' + str(i), fc_attr)

    def forward(self, x, lm):
        print(" initial ", x.size())
        x = self.features(x)
        print(x.size())
        print(" lm size ", lm.size())
        lt = lm
        # print(" unsq ", lm.unsqueeze(2).unsqueeze(3).size())

        # lm = F.interpolate(lm.unsqueeze(2).unsqueeze(3), scale_factor=2, mode='bicubic', align_corners=False)
        print(" interpol LM ", lm.size())
        gx = self.layer_next(x)
        print(" afte glayer ", gx.size())
        # gx = F.interpolate(gx, scale_factor=2, mode='bicubic', align_corners=False)
        # print(" interpol ", gx.size())
        # gx = self.upsample(gx)
        gx = self.avg_pooling(gx)

        gx = self.max(gx)
        print(" after pool input ", gx.size())
        gx = torch.flatten(gx, 1)

        lx = self.roi_fc(x)
        print(" roi lx ", lx.size())
        lm = self.down(lm.unsqueeze(2)).squeeze(2)
        print(" lm first conv ", lm.size())
        # lm = lm.squeeze(2)
        lm = self.uprelu(lm)
        # lm = F.interpolate(lm, scale_factor=2, mode="bicubic", align_corners = False)
        lm = self.up(lm.unsqueeze(2)).squeeze(2)
        print(" lm upsample ", lm.size())
        # lm = lm.squeeze(2)
        lx = self.roi_pooling_layer(lx, lm)
        print(" after roi pool ", lx.size())
        lx = torch.flatten(lx, 1)
        lx = self.llayer(lx)
        print(" last layer ", lx.size())
        # lx = self.up(lx.squeeze(1))
        lx = torch.flatten(lx, 1)

        fx = torch.cat((gx, lx), dim=1)
        print(" after concat ", fx.size())
        fx = self.fusion(fx)
        print(" after fuson ", fx.size())

        output = []
        for i in range(6):
            fc_attr = getattr(self, 'fc' + str(i))
            output.append(fc_attr(fx))
        return output

