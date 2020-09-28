# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from torch import nn


def get_nonlinearity_layer(nonlinearity_type, inplace):
    r"""Return a nonlinearity layer.

    Args:
        nonlinearity_type (str):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace (bool): If ``True``, set ``inplace=True`` when initializing
            the nonlinearity layer.
    """
    if nonlinearity_type == 'relu':
        nonlinearity = nn.ReLU(inplace=inplace)
    elif nonlinearity_type == 'leakyrelu':
        nonlinearity = nn.LeakyReLU(0.2, inplace=inplace)
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
        raise ValueError('Nonlinearity %s is not recognized' %
                         nonlinearity_type)
    return nonlinearity
