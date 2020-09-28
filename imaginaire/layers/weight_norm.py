# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools

import torch
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

from .conv import LinearBlock


class WeightDemodulation(nn.Module):
    r"""Weight demodulation in
    "Analyzing and Improving the Image Quality of StyleGAN", Karras et al.

    Args:
        conv (torch.nn.Modules): Convolutional layer.
        cond_dims (int): The number of channels in the conditional input.
        eps (float, optional, default=1e-8): a value added to the
            denominator for numerical stability.
        adaptive_bias (bool, optional, default=False): If ``True``, adaptively
            predicts bias from the conditional input.
        demod (bool, optional, default=False): If ``True``, performs
            weight demodulation.
    """

    def __init__(self, conv, cond_dims, eps=1e-8,
                 adaptive_bias=False, demod=True):
        super().__init__()
        self.conv = conv
        self.adaptive_bias = adaptive_bias
        if adaptive_bias:
            self.conv.register_parameter('bias', None)
            self.fc_beta = LinearBlock(cond_dims, self.conv.out_channels)
        self.fc_gamma = LinearBlock(cond_dims, self.conv.in_channels)
        self.eps = eps
        self.demod = demod
        self.conditional = True

    def forward(self, x, y):
        r"""Weight demodulation forward"""
        b, c, h, w = x.size()
        self.conv.groups = b
        gamma = self.fc_gamma(y)
        gamma = gamma[:, None, :, None, None]
        weight = self.conv.weight[None, :, :, :, :] * (gamma + 1)

        if self.demod:
            d = torch.rsqrt(
                (weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d

        x = x.reshape(1, -1, h, w)
        _, _, *ws = weight.shape
        weight = weight.reshape(b * self.conv.out_channels, *ws)
        x = self.conv.conv2d_forward(x, weight)

        x = x.reshape(-1, self.conv.out_channels, h, w)
        if self.adaptive_bias:
            x += self.fc_beta(y)[:, :, None, None]
        return x


def weight_demod(conv, cond_dims=256, eps=1e-8, demod=True):
    r"""Weight demodulation."""
    return WeightDemodulation(conv, cond_dims, eps, demod)


def get_weight_norm_layer(norm_type, **norm_params):
    r"""Return weight normalization.

    Args:
        norm_type (str):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        norm_params: Arbitrary keyword arguments that will be used to
            initialize the weight normalization.
    """
    if norm_type == 'none' or norm_type == '':  # no normalization
        return lambda x: x
    elif norm_type == 'spectral':  # spectral normalization
        return functools.partial(spectral_norm, **norm_params)
    elif norm_type == 'weight':  # weight normalization
        return functools.partial(weight_norm, **norm_params)
    elif norm_type == 'weight_demod':  # weight demodulation
        return functools.partial(weight_demod, **norm_params)
    else:
        raise ValueError(
            'Weight norm layer %s is not recognized' % norm_type)
