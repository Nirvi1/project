# -*- coding:utf-8 -*-
"""
File: helper.py
File Created: 2021-09-20
Author: Nirvi Badyal
"""
import math
from torch.autograd import Variable

import torch.nn.functional as F
from src.focal_loss import FocalLossFunction
import torch
from torch import nn
LIST_ATTR = [7, 3, 3, 4, 6, 3]
LIST_W_ATTR = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]


def grad_set(model, req=False):
    for param in model.parameters():
        param.requires_grad = req


def get_accuracy(output, target):
    correct = []
    for i in range(6):
        pred = output[i].argmax(dim=1, keepdim=True)
        correct.append(pred.eq(target[:, i].view_as(pred)).sum().item())
    sum_correct = sum(correct)
    return sum_correct, correct


def get_standard_cross_entropy_loss(output, target):
    loss = []
    for i in range(6):
        #focal_obj = FocalLossFunction()
        #focal_loss = focal_obj(output[i], target[:, i])
        #loss.append(focal_loss)  
        loss.append(F.cross_entropy(output[i], target[:, i]))
       
    sum_loss = sum([LIST_W_ATTR[i] * loss[i] for i in range(6)])
    return sum_loss, loss

def get_focal_loss(output, target, gamma=2):
    loss = []
    for i in range(6):
        fl_func = FocalLossFunction(class_num=LIST_ATTR[i], gamma=gamma)
        loss.append(fl_func.forward(output[i], target[:, i]))
    sum_loss = sum([LIST_W_ATTR[i] * loss[i] for i in range(6)])
    return sum_loss, loss

def get_loss_Triplet(output, target, gamma=2):
    loss = []
    for i in range(6):
        fl_func = TripletLoss(margin=0.3, loss_weight=1.0)
        loss.append(fl_func.forward(output[i], target[:, i]))
    sum_loss = sum([LIST_W_ATTR[i] * loss[i] for i in range(6)])
    return sum_loss, loss

class FocalLossFunction(nn.Module):
    r"""
        Implemenation of Focal Loss, which is proposed in paper
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each batch of 64 size.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLossFunction, self).__init__()

        self.alpha = alpha if alpha is not None else 0.6
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        masker = inputs.data.new(N, C).fill_(0)
        masker = Variable(masker)
        ids = targets.view(-1, 1)
        masker.scatter_(1, ids.data, 1.)

        probs = (P * masker).sum(1).view(-1, 1)

        log_p = probs.log()

        loss_b = -self.alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = loss_b.mean()
        else:
            loss = loss_b.sum()
        return loss

class RoIPoolNetwork(nn.Module):

    def __init__(self, output, bbox, scale):
        super(RoIPoolNetwork, self).__init__()
        self.output = output
        self.bbox = bbox
        self.scale = scale

    def mapping(self, i):
        h1 = self.output[0] * (i // 2)
        h2 = h1 + self.output[0]
        w1 = self.output[1] * (i % 2)
        w2 = w1 + self.output[1]
        return h1, h2, w1, w2

    def forward(self, input, lm):
        batch_size, input_channel = input.size()[0], input.size()[1]
        output = input.new_zeros(
            [batch_size, input_channel, 
             self.output[0] * 4, self.output[1] * 2])
        for i in range(8):
            # build roi window
            x0, y0 = lm[:, 2*i], lm[:, 2*i+1]
            visible = (x0 > 0).to(int) | (y0 > 0).to(int)

            x1 = torch.floor(x0 * self.scale) - math.floor(self.bbox[0] / 2)
            x2 = x1 + math.ceil(self.bbox[0] / 2)
            y1 = torch.floor(y0 * self.scale) - math.floor(self.bbox[1] / 2)
            y2 = y1 + math.ceil(self.bbox[1] / 2)
            rois = torch.stack(
                [torch.arange(batch_size, device=input.device), x1, x2, y1, y2], dim=1)
            
            out_pool, _ = torch.ops.torchvision.roi_pool(input, rois, 1.0,
                self.output[0], self.output[1])
            # out_pool = torch.flatten(output, 1)
            out_pool = (visible * out_pool.T).T
            h1, h2, w1, w2 = self.mapping(i)
            output[:, :, h1:h2, w1:w2] = out_pool
        return output

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output=' + str(self.output)
        tmpstr += ', scale=' + str(self.scale)
        tmpstr += ')'
        return tmpstr

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Args:
        margin (float, optional): Margin for triplet loss. Default to 0.3.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
    """

    def __init__(self, margin=0.3, loss_weight=1.0, hard_mining=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.loss_weight = loss_weight
        self.hard_mining = hard_mining

    def hard_mine_forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (num_classes).
        """

        batch_size = inputs.size(0)

        # Compute Euclidean distance
        dist = torch.pow(inputs, 2).sum(
            dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the furthest positive sample
        # and nearest negative sample in the embedding space
        mask = targets.expand(batch_size, batch_size).eq(
            targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.loss_weight * self.ranking_loss(dist_an, dist_ap, y)

    def forward(self, inputs, targets, **kwargs):
        if self.hard_mining:
            return self.hard_mine_forward(inputs, targets)
        else:
            raise NotImplementedError()
