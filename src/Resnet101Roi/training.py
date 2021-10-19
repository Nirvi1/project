# -*- coding:utf-8 -*-
"""
File: training.py
File Created: 2021-09-20
Author: Nirvi Badyal
"""
import torch
from torch.autograd import Variable

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import get_standard_cross_entropy_loss, get_accuracy, get_focal_loss, get_loss_Triplet


def train(model, training_data_loader, optimizer, verbose=0, device='cpu'):
    if device == 'cpu':
        model.cpu()
    else:
        model.cuda()
    model.train()
    running_loss = 0.0
    running_corrects, running_total = 0, 0
    
    for d in training_data_loader:
        if device == 'cpu':
            inputs, lm, targets = Variable(d['img']), Variable(d['landmark']), Variable(d['attr'])
        else:
            inputs, lm = Variable(d['img']).cuda(), Variable(d['landmark']).cuda()
            targets = Variable(d['attr']).cuda()

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss, loss_lst = get_standard_cross_entropy_loss(outputs, targets)
            corr, corr_lst = get_accuracy(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += corr
        running_total += inputs.size(0)
        if verbose >= 3:
            acc = corr / 6 / inputs.size(0)
            print(f'  Batch Train:  Loss: {loss:.4f},  Accuracy:{acc:.4f}')

    epoch_loss = running_loss / running_total
    epoch_acc = float(running_corrects) / 6 / running_total
    return epoch_loss, epoch_acc

def train_roi(model, training_data_loader, optimizer, verbose=0, device='cpu', cross=False, triplet_loss = False):
    if device == 'cpu':
        model.cpu()
    else:
        model.cuda()
    model.train()
    running_loss = 0.0
    running_corrects, running_total = 0, 0
    
    for d in training_data_loader:
        if device == 'cpu':
            inputs, lm, targets = Variable(d['img']), Variable(d['landmark']), Variable(d['attr'])
        else:
            inputs, lm = Variable(d['img']).cuda(), Variable(d['landmark']).cuda()
            targets = Variable(d['attr']).cuda()

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs, lm)
            if cross == False:
              loss, loss_lst = get_focal_loss(outputs, targets)
            elif triplet_loss == True:
              loss, loss_lst = get_loss_Triplet(outputs, targets)
            else:
              loss, loss_lst = get_standard_cross_entropy_loss(outputs, targets)
            corr, corr_lst = get_accuracy(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += corr
        running_total += inputs.size(0)
        if verbose >= 3:
            acc = corr / 6 / inputs.size(0)
            # print(f'  Batch Train:  Loss: {loss:.4f},  Accuracy:{acc:.4f}')
            print('  Batch Train:  Loss: {:.4f},  Accuracy:{:.4f}'.format(loss, acc))

    epoch_loss = running_loss / running_total
    epoch_acc = float(running_corrects) / 6 / running_total
    return epoch_loss, epoch_acc

def eval(model, validation_loader, optimizer, verbose=0, device='cpu', cross=False, triplet_loss=False):
    if device == 'cpu':
        model.cpu()
    else:
        model.cuda()
    model.eval()
    running_loss = 0.0
    running_corrects, running_total = 0, 0
    
    for d in validation_loader:
        if device == 'cpu':
            inputs, lm, targets = Variable(d['img']), Variable(d['landmark']), Variable(d['attr'])
        else:
            inputs, lm = Variable(d['img']).cuda(), Variable(d['landmark']).cuda()
            targets = Variable(d['attr']).cuda()

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs, lm)
            if cross==False:
              loss, loss_lst = get_focal_loss(outputs, targets)
            elif triplet_loss == True:
              loss, loss_lst = get_loss_Triplet(outputs, targets)
            else:
              loss, loss_lst = get_standard_cross_entropy_loss(outputs, targets)
            corr, corr_lst = get_accuracy(outputs, targets)
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += corr
        running_total += inputs.size(0)
        if verbose >= 3:
            acc = corr / 6 / inputs.size(0)
            print(f'  Batch Test:  Loss: {loss:.4f},  Accuracy:{acc:.4f}')

    epoch_loss = running_loss / running_total
    epoch_acc = float(running_corrects) / 6 / running_total
    return epoch_loss, epoch_acc

def test_roi(model, dl_tes, verbose=0, device='cpu'):
    if device == 'cpu':
        model.cpu()
    else:
        model.cuda()
    model.eval()

    ans = torch.Tensor([]).to(int)
    for d in dl_tes:
        if device == 'cpu':
            inputs, lm = Variable(d['img']), Variable(d['landmark'])
        else:
            inputs, lm = Variable(d['img']).cuda(), Variable(d['landmark']).cuda()

        with torch.set_grad_enabled(False):
            outputs = model(inputs, lm)
            pred = [0] * 6
            for i in range(6):
                pred[i] = outputs[i].argmax(dim=1, keepdim=True)
            t = torch.cat(pred, dim=1)
            ans = torch.cat([ans, t], dim=0)

    return ans
