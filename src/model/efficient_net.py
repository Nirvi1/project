import timm

import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import math
import torchvision.transforms as T
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from collections import defaultdict
from bisect import bisect_right
import copy

class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s = 10, m = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)
        
        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
                
        model = timm.create_model('efficientnet_b3', pretrained=True)
        model.classifier = torch.nn.Identity()
        self.model = model
        self.margin = ArcModule(in_features=1536, out_features = 1806)
        
    def forward(self, img, labels=None):        
        global_feas = self.model(img)
        feat = F.normalize(global_feas)
        if labels is not None:
            return global_feas, self.margin(feat, labels)
        return feat
    
#model = EfficientNet().cuda()
