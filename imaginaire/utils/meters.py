# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import math
from datetime import timedelta

import torch
import wandb
from wandb import AlertLevel
from torch.utils.tensorboard import SummaryWriter

from imaginaire.utils.distributed import master_only, dist_all_reduce_tensor, \
    is_master, get_rank

from imaginaire.utils.distributed import master_only_print as print

LOG_WRITER = None
LOG_DIR = None


@torch.no_grad()
def sn_reshape_weight_to_matrix(weight):
    r"""Reshape weight to obtain the matrix form.

    Args:
        weight (Parameters): pytorch layer parameter tensor.
    """
    weight_mat = weight
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


@torch.no_grad()
def get_weight_stats(mod):
    r"""Get weight state

    Args:
         mod: Pytorch module
    """
    if mod.weight_orig.grad is not None:
        grad_norm = mod.weight_orig.grad.data.norm().item()
    else:
        grad_norm = 0.
    weight_norm = mod.weight_orig.data.norm().item()
    weight_mat = sn_reshape_weight_to_matrix(mod.weight_orig)
    sigma = torch.sum(mod.weight_u * torch.mv(weight_mat, mod.weight_v))
    return grad_norm, weight_norm, sigma


@master_only
def set_summary_writer(log_dir):
    r"""Set summary writer

    Args:
        log_dir (str): Log directory.
    """
    global LOG_DIR, LOG_WRITER
    LOG_DIR = log_dir
    LOG_WRITER = SummaryWriter(log_dir=log_dir)


def write_summary(name, summary, step, hist=False):
    """Utility function for write summary to log_writer.
    """
    global LOG_WRITER
    lw = LOG_WRITER
    if lw is None:
        raise Exception("Log writer not set.")
    if hist:
        lw.add_histogram(name, summary, step)
    else:
        lw.add_scalar(name, summary, step)


class Meter(object):
    """Meter is to keep track of statistics along steps.
    Meters write values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard for now)
    regularly.

    Args:
        name (str): the name of meter
        reduce (bool): If ``True``, perform a distributed reduce for the log
            values across all GPUs.
    """

    def __init__(self, name, reduce=True):
        self.name = name
        self.reduce = reduce
        self.values = []

    def reset(self):
        r"""Reset the meter values"""
        if not self.reduce and get_rank() != 0:
            return
        self.values = []

    def write(self, value):
        r"""Record the value"""
        if not self.reduce and get_rank() != 0:
            return
        if value is not None:
            self.values.append(value)

    def flush(self, step):
        r"""Write the value in the tensorboard.

        Args:
            step (int): Epoch or iteration number.
        """
        if not self.reduce and get_rank() != 0:
            return
        values = torch.tensor(self.values, device="cuda")
        if self.reduce:
            values = dist_all_reduce_tensor(values)

        if not all(math.isfinite(x) for x in values):
            print("meter {} contained a nan or inf.".format(self.name))
            if is_master():
                wandb.alert(
                    title='NaN',
                    text=f'Meter {self.name} contained a nan or inf.',
                    level=AlertLevel.WARN,
                    wait_duration=timedelta(minutes=120)
                )
        filtered_values = list(filter(lambda x: math.isfinite(x), self.values))
        if float(len(filtered_values)) != 0:
            value = float(sum(filtered_values)) / float(len(filtered_values))
            if is_master():
                write_summary(self.name, value, step)
                wandb.log({self.name: value}, step=step)
        self.reset()

    @master_only
    def write_image(self, img_grid, step):
        r"""Write the value in the tensorboard.

        Args:
            img_grid:
            step (int): Epoch or iteration number.
        """
        if not self.reduce and get_rank() != 0:
            return
        global LOG_WRITER
        lw = LOG_WRITER
        if lw is None:
            raise Exception("Log writer not set.")
        lw.add_image("Visualizations", img_grid, step)
