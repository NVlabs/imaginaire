# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    r"""Compute Weighted MSE loss"""
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight):
        r"""Return weighted MSE Loss.
        Args:
           input (tensor):
           target (tensor):
           weight (tensor):
        Returns:
           (tensor): Loss value.
        """
        if self.reduction == 'mean':
            loss = torch.mean(weight * (input - target) ** 2)
        else:
            loss = torch.sum(weight * (input - target) ** 2)
        return loss
