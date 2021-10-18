# -*- coding:utf-8 -*-
"""Draw line plots
File: pltplot.py
File Created: 2021-04-10
Author: Nirvi Badyal
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_dual(los_tra, los_val, acc_tra, acc_val):
    """
    Plot dual lines of loss and accuracy.

    Args:
        los_lst (list of floats): loss in epochs
        acc_lst (list of floats (range 0~100)): accuracy in epochs

    Returns:
        fig: figure object
    """
    fig, ax_acc = plt.subplots(figsize=(8, 6))

    cl_acc = 'tab:red'
    ax_acc.plot(acc_tra, '--', color=cl_acc)
    ax_acc.plot(acc_val, '-', color=cl_acc)
    ax_acc.set_xlabel('round')
    ax_acc.set_xlim([0, len(los_tra)])
    ax_acc.set_ylabel('Test Acc', color=cl_acc)
    ax_acc.set_ylim([0, 100])
    ax_acc.tick_params(axis='y', labelcolor=cl_acc)
    ax_acc.grid(True, ls='--')
    # ax_omg.set_title('$\\theta$ and $\omega$ over time')

    ax_los = ax_acc.twinx()
    cl_los = 'tab:blue'
    ax_los.plot(los_tra, '--', color=cl_los)
    ax_los.plot(los_val, '-', color=cl_los)
    ax_los.set_ylabel('Test Loss', color=cl_los)
    ax_los.set_ylim([0, max(los_tra)])
    ax_los.tick_params(axis='y', labelcolor=cl_los)
    ax_los.grid(False)

    fig.tight_layout()
    # plt.show()
    return fig
