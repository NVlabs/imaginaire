# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
import torch.nn as nn


class GaussianKLLoss(nn.Module):
    r"""Compute KL loss in VAE for Gaussian distributions"""
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu, logvar=None):
        r"""Compute loss

        Args:
            mu (tensor): mean
            logvar (tensor): logarithm of variance
        """
        if logvar is None:
            logvar = torch.zeros_like(mu)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
