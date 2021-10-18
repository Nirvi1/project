# -*- coding:utf-8 -*-
"""Statistical tools for parameter counters, etc.
File: statistic.py
File Created: 2021-04-25
Author: Nirvi Badyal
"""
from collections import Iterable, OrderedDict
import torch


def numel_tensor(tensor_lst):
    """
    Parameter number counter.

    Args:
        tensor_lst (Iterable of tensors): tensor objects

    Returns:
        (int): parameters number
    """
    # assert isinstance(tensor_lst, Iterable), TypeError(
    #     '<{}> not supported'.format(type(tensor_lst)))
    # assert isinstance(list(tensor_lst)[0], torch.Tensor), TypeError(
    #     '<{}> not a tensor'.format(type(list(tensor_lst)[0])))
    return sum([int(t.numel()) for t in tensor_lst])


def numel_module(module_dict):
    """
    Parameter number counter.

    Args:
        module_dict (dict): state dict from nn.Module 

    Returns:
        (int): parameters number
    """
    # assert isinstance(module_dict, dict), TypeError(
    #     '<{}> not supported'.format(type(module_dict)))
    # assert isinstance(module_dict[list(module_dict)[0]], torch.Tensor), TypeError(
    #     '<{}> not a tensor'.format(type(module_dict[list(module_dict)[0]])))
    return sum([int(t.numel()) for t in module_dict.values()])


def nonzero_tensor(tensor_lst):
    """
    Non-zero parameter number counter.

    Args:
        tensor_lst (Iterable of tensors): tensor objects

    Returns:
        (int): non-zero parameters number
    """
    return sum([int(torch.sum(t != 0)) for t in tensor_lst])


def nonzero_module(module_dict):
    """
    Non-zero parameter number counter.

    Args:
        module_dict (dict): state dict from nn.Module 

    Returns:
        (int): non-zero parameters number
    """
    return sum([int(torch.sum(t != 0)) for t in module_dict.values()])


def numzero_tensor(tensor_lst):
    """
    Zero parameter number counter.

    Args:
        tensor_lst (Iterable of tensors): tensor objects

    Returns:
        (int): zero parameters number
    """
    return sum([int(torch.sum(t == 0)) for t in tensor_lst])


def numzero_module(module_dict):
    """
    Zero parameter number counter.

    Args:
        module_dict (dict): state dict from nn.Module 

    Returns:
        (int): zero parameters number
    """
    return sum([int(torch.sum(t == 0)) for t in module_dict.values()])
