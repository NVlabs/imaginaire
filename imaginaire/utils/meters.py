# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from apex import amp
from imaginaire.utils.distributed import master_only
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
def get_weight_stats(mod, cfg, loss_id):
    r"""Get weight state

    Args:
         mod: Pytorch module
         cfg: Configuration object
         loss_id: Needed when using AMP.
    """
    loss_scale = 1.0
    if cfg.trainer.amp == 'O1' or cfg.trainer.amp == 'O2':
        # AMP rescales the gradient so we have to undo it.
        loss_scale = amp._amp_state.loss_scalers[loss_id].loss_scale()
    if mod.weight_orig.grad is not None:
        grad_norm = mod.weight_orig.grad.data.norm().item() / float(loss_scale)
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


@master_only
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


@master_only
def add_hparams(hparam_dict=None, metric_dict=None):
    r"""Add a set of hyperparameters to be compared in tensorboard.

    Args:
        hparam_dict (dictionary): Each key-value pair in the dictionary is the
            name of the hyper parameter and it's corresponding value.
            The type of the value can be one of `bool`, `string`, `float`,
            `int`, or `None`.
        metric_dict (dictionary): Each key-value pair in the dictionary is the
            name of the metric and it's corresponding value. Note that the key
            used here should be unique in the tensorboard record. Otherwise the
            value you added by `add_scalar` will be displayed in hparam plugin.
            In most cases, this is unwanted.
    """
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError('hparam_dict and metric_dict should be dictionary.')
    global LOG_WRITER
    lw = LOG_WRITER

    exp, ssi, sei = hparams(hparam_dict, metric_dict)

    lw.file_writer.add_summary(exp)
    lw.file_writer.add_summary(ssi)
    lw.file_writer.add_summary(sei)


class Meter(object):
    """Meter is to keep track of statistics along steps.
    Meters write values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard for now)
    regularly.

    Args:
        name (str): the name of meter
    """

    @master_only
    def __init__(self, name):
        self.name = name
        self.values = []

    @master_only
    def reset(self):
        r"""Reset the meter values"""
        self.values = []

    @master_only
    def write(self, value):
        r"""Record the value"""
        self.values.append(value)

    @master_only
    def flush(self, step):
        r"""Write the value in the tensorboard.

        Args:
            step (int): Epoch or iteration number.
        """
        if not all(math.isfinite(x) for x in self.values):
            print("meter {} contained a nan or inf.".format(self.name))
        filtered_values = list(filter(lambda x: math.isfinite(x), self.values))
        if float(len(filtered_values)) != 0:
            value = float(sum(filtered_values)) / float(len(filtered_values))
            write_summary(self.name, value, step)
        self.reset()

    @master_only
    def write_image(self, img_grid, step):
        r"""Write the value in the tensorboard.

        Args:
            img_grid:
            step (int): Epoch or iteration number.
        """
        global LOG_WRITER
        lw = LOG_WRITER
        if lw is None:
            raise Exception("Log writer not set.")
        lw.add_image("Visualizations", img_grid, step)
