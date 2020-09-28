# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial

import torch
import torch.nn as nn

from imaginaire.layers import Conv2dBlock


class NonLocal2dBlock(nn.Module):
    r"""Self attention Layer

    Args:
        in_channels (int): Number of channels in the input tensor.
        scale (bool, optional, default=True): If ``True``, scale the
            output by a learnable parameter.
        clamp (bool, optional, default=``False``): If ``True``, clamp the
            scaling parameter to (-1, 1).
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
    """

    def __init__(self,
                 in_channels,
                 scale=True,
                 clamp=False,
                 weight_norm_type='none'):
        super(NonLocal2dBlock, self).__init__()
        self.clamp = clamp
        self.gamma = nn.Parameter(torch.zeros(1)) if scale else 1.0
        self.in_channels = in_channels
        base_conv2d_block = partial(Conv2dBlock,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    weight_norm_type=weight_norm_type)
        self.theta = base_conv2d_block(in_channels, in_channels // 8)
        self.phi = base_conv2d_block(in_channels, in_channels // 8)
        self.g = base_conv2d_block(in_channels, in_channels // 2)
        self.out_conv = base_conv2d_block(in_channels // 2, in_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        r"""

        Args:
            x (tensor) : input feature maps (B X C X W X H)
        Returns:
            (tuple):
              - out (tensor) : self attention value + input feature
              - attention (tensor): B x N x N (N is Width*Height)
        """
        n, c, h, w = x.size()
        theta = self.theta(x).view(n, -1, h * w).permute(0, 2, 1)

        phi = self.phi(x)
        phi = self.max_pool(phi).view(n, -1, h * w // 4)

        energy = torch.bmm(theta, phi)
        attention = self.softmax(energy)

        g = self.g(x)
        g = self.max_pool(g).view(n, -1, h * w // 4)

        out = torch.bmm(g, attention.permute(0, 2, 1))
        out = out.view(n, c // 2, h, w)
        out = self.out_conv(out)

        if self.clamp:
            out = self.gamma.clamp(-1, 1) * out + x
        else:
            out = self.gamma * out + x
        return out
