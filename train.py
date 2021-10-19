#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import sys

#cwd = os.getcwd()
#pwd = cwd[:cwd.rfind('/')]
#sys.path.append(pwd)

import numpy as np

import torch
from torch import optim

from src.data import get_data_tra, get_data_val, PATH_TO_DATA
from src.model import ResNet18Model, get_standard_cross_entropy_loss, get_accuracy
from src.utils.config import generate_settings, PATH_TO_SRC
from src.utils.logger import Logger, ModelNetworkLogger, PATH_TO_SAVE
from src.utils.scheduler import WarmingUpScheduler
from src.Resnet18.training import  eval
from torch.autograd import Variable


# In[2]:


device = 'cpu'
PATH_DATA = PATH_TO_DATA
PATH_SRC = PATH_TO_SRC
PATH_SAVE = PATH_TO_SAVE

prj_name = 'Resnet50'
verbose = 3
opt = generate_settings(prj_path=PATH_SRC, prj_name=prj_name)

flag_run = opt.flag_run
logger = Logger(prj_name, save_path=PATH_SAVE, flag_run=flag_run)
logger.print(" START ")
if not logger.path_existed:
    # newly run
    opt.flag_run = logger.flag_run
    logger.save_settings(opt)
else:
    # continue from checkpoint
    opt = logger.load_settings()
    opt.device = device


# In[3]:


# print('Initialize Model...')
model = ResNet18Model()
optimizer = optim.Adam(model.parameters(),
    lr=opt.optim.lr,
    weight_decay=opt.optim.wd)
scheduler_after = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=opt.optim.milestones,
    gamma=opt.optim.gamma)
scheduler = WarmingUpScheduler(
    optimizer,
    multiplier=1.0,
    total_epoch=opt.optim.warmup,
    after_scheduler=scheduler_after)

model_logger = ModelNetworkLogger(logger, prefix='model')
model_logger.add_trained_model(model, save_init=False)
model_logger.model_state_load(optimizer=optimizer, scheduler=scheduler, save_init=False)

if not logger.path_existed:
    curr_epoch = 0
else:
    curr_state = model_logger.load_state('model', optimizer=optimizer, scheduler=scheduler)
    curr_epoch = curr_state['epoch']
    optimizer = curr_state['optimizer']
    scheduler = curr_state['scheduler']
    model = model_logger.model_load(str(curr_round), model=model)


# In[6]:


print('Initialize Dataset...')
training_data_loader = get_data_tra(
    data_path=PATH_DATA, 
    batch_size=opt.data.training_batch_size,
    img_size=opt.data.img_size)
validation_loader = get_data_val(
    data_path=PATH_DATA, 
    batch_size=opt.data.test_batch_size,
    img_size=opt.data.img_size)


# In[ ]:


def train(model, training_data_loader, optimizer, verbose=0, device='cpu'):
    if device == 'cpu':
        model.cpu()
    else:
        model.cuda()
    model.train()
    running_loss = 0.0
    running_corrects, running_total = 0, 0
    i = 0
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
            print(f'  Batch Train:  Loss: {loss:.4f},  Acc: {acc:.4f}')

    epoch_loss = running_loss / running_total
    epoch_acc = float(running_corrects) / 6 / running_total
    return epoch_loss, epoch_acc
logger.print(" START TRAINING ")
for epoch in range(curr_epoch, opt.optim.epochs):
    loss_tra, acc_tra = train(model, training_data_loader, optimizer, verbose, device)
    logger.print(f'Epoch {epoch:2d} Train:  Loss: {loss_tra:.4f},  Acc: {acc_tra:.4f}')
    print('Saving Model....')
    torch.save(model.state_dict(), '/ml_model/second/DeepFashion/models/model_' +str(epoch) )
    print('OK.')
    if scheduler is not None:
        scheduler.step()

    loss_val, acc_val = eval(model, validation_loader, optimizer, verbose, device)
    logger.print(f'Epoch {epoch:2d} Val:  Loss: {loss_val:.4f},  Acc: {acc_val:.4f}')
    model_logger.save_epoch(epoch=epoch, period=opt.model.period)
    model_logger.save_best(epoch=epoch, acc_curr=acc_val)





