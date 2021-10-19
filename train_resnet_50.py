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
from src.utils.logger import Logger, ModelLogger, PATH_TO_SAVE
from src.utils.scheduler import GradualWarmupScheduler
from src.Resnet18.run import  eval
from torch.autograd import Variable

import torch.optim.lr_scheduler as lr_scheduler

# In[2]:


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
    logger.save_opt(opt)
else:
    # continue from checkpoint
    opt = logger.load_opt()
    opt.device = device


# In[3]:


# print('Initialize Model...')
model = ResNet18Model()
optimizer = optim.Adam(model.parameters(),
    lr=opt.optim.lr, betas=(opt.optim.momentum, 0.99))
scheduler = lr_scheduler.StepLR(optimizer, opt.optim.lr_decay_step, opt.optim.lr_decay_rate)
model_logger = ModelLogger(logger, prefix='model')
model_logger.regi_model(model, save_init=False)
model_logger.regi_state(optimizer=optimizer, scheduler=scheduler, save_init=False)

if not logger.path_existed:
    curr_epoch = 0
else:
    curr_state = model_logger.load_state('model', optimizer=optimizer, scheduler=scheduler)
    curr_epoch = curr_state['epoch']
    optimizer = curr_state['optimizer']
    scheduler = curr_state['scheduler']
    model = model_logger.load_model(str(curr_round), model=model)


# In[6]:


print('Initialize Dataset...')
dl_tra = get_data_tra(
    data_path=PATH_DATA, 
    batch_size=opt.data.bs_tra,
    img_size=opt.data.img_size)
dl_val = get_data_val(
    data_path=PATH_DATA, 
    batch_size=opt.data.bs_tes,
    img_size=opt.data.img_size)


# In[ ]:


def train(model, dl_tra, optimizer, verbose=0, device='cpu'):
    if device == 'cpu':
        model.cpu()
    else:
        model.cuda()
    model.train()
    running_loss = 0.0
    running_corrects, running_total = 0, 0
    i = 0
    for d in dl_tra:
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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for epoch in range(curr_epoch, opt.optim.epochs):
    loss_tra, acc_tra = train(model, dl_tra, optimizer, verbose, device)
    logger.print(f'Epoch {epoch:2d} Train:  Loss: {loss_tra:.4f},  Acc: {acc_tra:.4f}')
    print('Saving Model....')
    torch.save(model.state_dict(), '/ml_model/second/DeepFashion/models/model_' +str(epoch) )
    print('OK.')
    if scheduler is not None:
        scheduler.step()

    loss_val, acc_val = eval(model, dl_val, optimizer, verbose, device)
    logger.print(f'Epoch {epoch:2d} Val:  Loss: {loss_val:.4f},  Acc: {acc_val:.4f}')
    model_logger.save_epoch(epoch=epoch, period=opt.model.period)
    model_logger.save_best(epoch=epoch, acc_curr=acc_val)





