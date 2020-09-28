# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import warnings

import torch
import torch.nn as nn

from imaginaire.layers import Conv2dBlock, Res2dBlock


class ResDiscriminator(nn.Module):
    r"""Global residual discriminator.

    Args:
        image_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        max_num_filters (int): Maximum num. of filters in a layer.
        first_kernel_size (int): Kernel size in the first layer.
        num_layers (int): Num. of layers in discriminator.
        padding_mode (str): Padding mode.
        activation_norm_type (str): Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        aggregation (str): Method to aggregate features across different
            locations in the final layer. ``'conv'``, or ``'pool'``.
        order (str): Order of operations in the residual link.
        anti_aliased (bool): If ``True``, uses anti-aliased pooling.
    """

    def __init__(self,
                 image_channels=3,
                 num_filters=64,
                 max_num_filters=512,
                 first_kernel_size=1,
                 num_layers=4,
                 padding_mode='zeros',
                 activation_norm_type='',
                 weight_norm_type='',
                 aggregation='conv',
                 order='pre_act',
                 anti_aliased=False,
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type' and key != 'patch_wise':
                warnings.warn(
                    "Discriminator argument {} is not used".format(key))

        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity='leakyrelu')

        first_padding = (first_kernel_size - 1) // 2
        model = [Conv2dBlock(image_channels, num_filters,
                             first_kernel_size, 1, first_padding,
                             **conv_params)]
        for _ in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            model.append(Res2dBlock(num_filters_prev, num_filters, order=order,
                                    **conv_params))
            model.append(nn.AvgPool2d(2, stride=2))
        if aggregation == 'pool':
            model += [torch.nn.AdaptiveAvgPool2d(1)]
        elif aggregation == 'conv':
            model += [Conv2dBlock(num_filters, num_filters, 4, 1, 0,
                                  nonlinearity='leakyrelu')]
        else:
            raise ValueError('The aggregation mode is not recognized'
                             % self.aggregation)
        self.model = nn.Sequential(*model)
        self.classifier = nn.Linear(num_filters, 1)

    def forward(self, images):
        r"""Multi-resolution patch discriminator forward.

        Args:
            images (tensor) : Input images.
        Returns:
            (tuple):
              - outputs (tensor): Output of the discriminator.
              - features (tensor): Intermediate features of the discriminator.
              - images (tensor): Input images.
        """
        batch_size = images.size(0)
        features = self.model(images)
        outputs = self.classifier(features.view(batch_size, -1))
        return outputs, features, images
