# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# import torch
import math

from torch.optim.optimizer import Optimizer, required


class Fromage(Optimizer):
    r"""Fromage optimizer implementation (https://arxiv.org/abs/2002.03432)"""

    def __init__(self, params, lr=required, momentum=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, momentum=momentum)
        super(Fromage, self).__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p_norm = p.grad.norm()
                p_norm = p.norm()
                if p_norm > 0.0 and d_p_norm > 0.0:
                    p.data.add_(-group['lr'], d_p * (p_norm / d_p_norm))
                else:
                    p.data.add_(-group['lr'], d_p)
                p.data /= math.sqrt(1 + group['lr'] ** 2)

        return loss
