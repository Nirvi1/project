import torch
import torch.nn as nn
from torchvision import models

from .helper import LIST_ATTR, grad_set, RoIPoolNetwork


class ResNet121Model(nn.Module):
    def __init__(self):
        super(ResNet121Model, self).__init__()

        backbone = models.densenet121(pretrained=True)
        backbone = list(backbone.children())
        self.features = nn.Sequential(*backbone[0][:-1])
        # grad_set(self.features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        fc_in = backbone[-1].in_features
        for i in range(6):
            fc_attr = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(
                    in_features=fc_in, out_features=LIST_ATTR[i]))
            setattr(self, 'fc' + str(i), fc_attr)

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        output = []
        for i in range(6):
          fc_attr = getattr(self, 'fc' + str(i))
          output.append(fc_attr(x))
        return output


class DensenetRoi(nn.Module):
    def __init__(self):
        super(DensenetRoi, self).__init__()

        backbone = models.densenet121(pretrained=True)
        backbone = list(backbone.children())
        self.features = nn.Sequential(*backbone[0][:-3])
        # grad_set(self.features)
        # DenseBlock3 output: [B, 1024, 14, 14]
        self.glayer = nn.Sequential(*backbone[0][-3:])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Transition3 ~ DenseBlock4 output: [B, 1024, 1, 1]

        self.roi_pool = RoIPoolNetwork(
            output=[2, 2],
            bbox=[1, 1],
            scale=14./224)
        # [B, 1024, 8, 4]
        self.roi_out = 256
        self.llayer = nn.Linear(in_features=1024*8*4,
                                out_features=self.roi_out)
        self.lact = nn.ReLU()
        
        fc_in = backbone[-1].in_features
        self.fusion = nn.Linear(in_features=fc_in + self.roi_out,
                                out_features=fc_in)
        
        for i in range(6):
            fc_attr = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(
                    in_features=fc_in, out_features=LIST_ATTR[i]))
            setattr(self, 'fc' + str(i), fc_attr)

    def forward(self, x, lm):
        x = self.features(x)
        # print(x.shape)

        gx = self.glayer(x)
        gx = self.avgpool(gx)
        gx = torch.flatten(gx, 1)

        lx = self.roi_pool(x, lm)
        lx = torch.flatten(lx, 1)
        lx = self.lact(self.llayer(lx))

        x = torch.cat((gx, lx), dim=1)
        x = self.fusion(x)
        
        output = []
        for i in range(6):
          fc_attr = getattr(self, 'fc' + str(i))
          output.append(fc_attr(x))
        return output
