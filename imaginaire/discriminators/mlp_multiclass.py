# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools

import numpy as np
import torch.nn as nn

from imaginaire.layers import LinearBlock


class Discriminator(nn.Module):
    r"""Multi-layer Perceptron Classifier constructor.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, dis_cfg, data_cfg):
        super(Discriminator, self).__init__()
        num_input_channels = dis_cfg.input_dims
        num_labels = dis_cfg.num_labels
        num_layers = getattr(dis_cfg, 'num_layers', 5)
        num_filters = getattr(dis_cfg, 'num_filters', 512)
        activation_norm_type = getattr(dis_cfg,
                                       'activation_norm_type',
                                       'batch_norm')
        nonlinearity = getattr(dis_cfg, 'nonlinearity', 'leakyrelu')
        base_linear_block = \
            functools.partial(LinearBlock,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity,
                              order='CNA')
        dropout_ratio = 0.1
        layers = [base_linear_block(num_input_channels, num_filters),
                  nn.Dropout(dropout_ratio)]
        for n in range(num_layers):
            dropout_ratio *= 1.5
            dropout_ratio = np.min([dropout_ratio, 0.5])
            layers += [base_linear_block(num_filters, num_filters),
                       nn.Dropout(dropout_ratio)]
        layers += [LinearBlock(num_filters, num_labels)]
        self.model = nn.Sequential(*layers)

    def forward(self, data):
        r"""Patch Discriminator forward.

        Args:
            data (dict):
              - data (N x -1 tensor): We will reshape the tensor to this format.
        Returns:
            (dict):
              - results (N x C tensor): Output scores before softmax.
        """
        input_x = data['data']
        bs = input_x.size()[0]
        input_x = input_x.view(bs, -1)
        pre_softmax_scores = self.model(input_x)
        outputs = dict()
        outputs['results'] = pre_softmax_scores
        return outputs
