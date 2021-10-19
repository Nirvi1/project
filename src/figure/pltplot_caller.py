# -*- coding:utf-8 -*-
"""Caller methods of `pltplot.py`. Can be called as module or run directly
File: pltplot_caller.py
File Created: 2021-04-09
Author: Nirvi Badyal
"""
import os
import argparse
import re

from .pltplot import plot_dual


def plot_dual_call(dir_save):
    """
    Call the `plot_dual` method base on given log path.

    Args:
        dir_save (str): saving dir
    """
    file_log = os.path.join(dir_save, 'log.txt')
    file_fig = os.path.join(dir_save, 'plot_dual.pdf')
    with open(file_log, 'r') as f:
        s = f.read()
    
    it_tra = re.finditer(
        r' Train:  Loss: (\d+\.\d+),  Accuracy:(\d+\.\d+)', s)
    loss_tra, acc_tra = [], []
    for i in it_tra:
        loss_tra.append(float(i.group(1)))
        acc_tra.append(100 * float(i.group(2)))

    it_val = re.finditer(
        r' Val:  Loss: (\d+\.\d+),  Accuracy:(\d+\.\d+)', s)
    loss_val, acc_val = [], []
    for i in it_val:
        loss_val.append(float(i.group(1)))
        acc_val.append(100 * float(i.group(2)))

    fig = plot_dual(loss_tra, loss_val, acc_tra, acc_val)
    fig.savefig(file_fig, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='dir path')
    parser.add_argument('-f', '--flag', type=str, help='run path')
    args = parser.parse_args()

    dir_save = os.path.join("../../save/", args.dir, args.flag)
    # dir_save = './src/figure'
    plot_dual_call(dir_save)
