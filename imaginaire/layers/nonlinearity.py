# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch import nn
import torch.nn.functional as F

from imaginaire.third_party.bias_act.bias_act import FusedNonlinearity


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2, scale=2 ** 0.5, inplace=False):
        super().__init__()

        self.negative_slope = negative_slope
        self.scale = scale
        self.inplace = inplace

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope, inplace=self.inplace) * self.scale
        # return _fused_scaled_leakyrelu(x, self.negative_slope, self.inplace, self.scale)


# @torch.jit.script
# def _fused_scaled_leakyrelu(x: torch.Tensor, negative_slope: float, inplace: bool, scale: float):
#     return F.leaky_relu(x, negative_slope, inplace=inplace) * scale


def get_nonlinearity_layer(nonlinearity_type, inplace, **kwargs):
    r"""Return a nonlinearity layer.

    Args:
        nonlinearity_type (str):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace (bool): If ``True``, set ``inplace=True`` when initializing
            the nonlinearity layer.
    """
    if nonlinearity_type.startswith('fused'):
        nonlinearity = FusedNonlinearity(nonlinearity=nonlinearity_type[6:], **kwargs)
    elif nonlinearity_type == 'relu':
        nonlinearity = nn.ReLU(inplace=inplace)
    elif nonlinearity_type == 'leakyrelu':
        nonlinearity = nn.LeakyReLU(0.2, inplace=inplace)
    elif nonlinearity_type == 'scaled_leakyrelu':
        nonlinearity = ScaledLeakyReLU(0.2, inplace=inplace)
        import imaginaire.config
        if imaginaire.config.USE_JIT:
            nonlinearity = torch.jit.script(nonlinearity)
    elif nonlinearity_type == 'prelu':
        nonlinearity = nn.PReLU()
    elif nonlinearity_type == 'tanh':
        nonlinearity = nn.Tanh()
    elif nonlinearity_type == 'sigmoid':
        nonlinearity = nn.Sigmoid()
    elif nonlinearity_type.startswith('softmax'):
        dim = nonlinearity_type.split(',')[1] if ',' in nonlinearity_type else 1
        nonlinearity = nn.Softmax(dim=int(dim))
    elif nonlinearity_type == 'none' or nonlinearity_type == '':
        nonlinearity = None
    else:
        raise ValueError('Nonlinearity %s is not recognized' % nonlinearity_type)
    return nonlinearity
