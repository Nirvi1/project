# -*- coding:utf-8 -*-
"""
File: run.py
File Created: 2021-09-20
Author: Nirvi Badyal
"""
import torch
from torch.autograd import Variable

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import get_loss_NCE, get_acc


def train(model, dl_tra, optimizer, verbose=0, device='cpu'):
    if device == 'cpu':
        model.cpu()
    else:
        model.cuda()
    model.train()
    running_loss = 0.0
    running_corrects, running_total = 0, 0
    
    for d in dl_tra:
        if device == 'cpu':
            inputs, lm, targets = Variable(d['img']), Variable(d['landmark']), Variable(d['attr'])
        else:
            inputs, lm = Variable(d['img']).cuda(), Variable(d['landmark']).cuda()
            targets = Variable(d['attr']).cuda()

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss, loss_lst = get_loss_NCE(outputs, targets)
            corr, corr_lst = get_acc(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += corr
        running_total += inputs.size(0)
        if verbose >= 3:
            acc = corr / 6 / inputs.size(0)
            print(f'  Batch Train:  Loss: {loss:.4f},  Acc: {acc:.4f}')

    epoch_loss = running_loss / running_total
    epoch_acc = float(running_corrects) / 6 / running_total
    return epoch_loss, epoch_acc


def eval(model, dl_val, optimizer, verbose=0, device='cpu'):
    if device == 'cpu':
        model.cpu()
    else:
        model.cuda()
    model.eval()
    running_loss = 0.0
    running_corrects, running_total = 0, 0
    
    for d in dl_val:
        if device == 'cpu':
            inputs, lm, targets = Variable(d['img']), Variable(d['landmark']), Variable(d['attr'])
        else:
            inputs, lm = Variable(d['img']).cuda(), Variable(d['landmark']).cuda()
            targets = Variable(d['attr']).cuda()

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss, loss_lst = get_loss_NCE(outputs, targets)
            corr, corr_lst = get_acc(outputs, targets)
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += corr
        running_total += inputs.size(0)
        if verbose >= 3:
            acc = corr / 6 / inputs.size(0)
            print(f'  Batch Test:  Loss: {loss:.4f},  Acc: {acc:.4f}')

    epoch_loss = running_loss / running_total
    epoch_acc = float(running_corrects) / 6 / running_total
    return epoch_loss, epoch_acc
