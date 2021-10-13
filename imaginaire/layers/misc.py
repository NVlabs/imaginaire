# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch import nn


class ApplyNoise(nn.Module):
    r"""Add Gaussian noise to the input tensor."""

    def __init__(self):
        super().__init__()
        # scale of the noise
        self.scale = nn.Parameter(torch.zeros(1))
        self.conditional = True

    def forward(self, x, *_args, noise=None, **_kwargs):
        r"""

        Args:
            x (tensor): Input tensor.
            noise (tensor, optional, default=``None``) : Noise tensor to be
                added to the input.
        """
        if noise is None:
            sz = x.size()
            noise = x.new_empty(sz[0], 1, *sz[2:]).normal_()

        return x + self.scale * noise


class PartialSequential(nn.Sequential):
    r"""Sequential block for partial convolutions."""
    def __init__(self, *modules):
        super(PartialSequential, self).__init__(*modules)

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensor.
        """
        act = x[:, :-1]
        mask = x[:, -1].unsqueeze(1)
        for module in self:
            act, mask = module(act, mask_in=mask)
        return act


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        if isinstance(size, int):
            h, w = size, size
        else:
            h, w = size
        self.input = nn.Parameter(torch.randn(1, channel, h, w))

    def forward(self):
        return self.input
