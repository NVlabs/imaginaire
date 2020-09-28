# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import functools
import warnings

import numpy as np
import torch
import torch.nn as nn

from imaginaire.layers import Conv2dBlock
from imaginaire.utils.data import (get_paired_input_image_channel_number,
                                   get_paired_input_label_channel_number)
from imaginaire.utils.distributed import master_only_print as print


class Discriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config
            file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, dis_cfg, data_cfg):
        super(Discriminator, self).__init__()
        print('Multi-resolution patch discriminator initialization.')
        # We assume the first datum is the ground truth image.
        image_channels = get_paired_input_image_channel_number(data_cfg)
        # Calculate number of channels in the input label.
        num_labels = get_paired_input_label_channel_number(data_cfg)

        # Build the discriminator.
        kernel_size = getattr(dis_cfg, 'kernel_size', 3)
        num_filters = getattr(dis_cfg, 'num_filters', 128)
        max_num_filters = getattr(dis_cfg, 'max_num_filters', 512)
        num_discriminators = getattr(dis_cfg, 'num_discriminators', 2)
        num_layers = getattr(dis_cfg, 'num_layers', 5)
        activation_norm_type = getattr(dis_cfg, 'activation_norm_type', 'none')
        weight_norm_type = getattr(dis_cfg, 'weight_norm_type', 'spectral')
        print('\tBase filter number: %d' % num_filters)
        print('\tNumber of discriminators: %d' % num_discriminators)
        print('\tNumber of layers in a discriminator: %d' % num_layers)
        print('\tWeight norm type: %s' % weight_norm_type)
        num_input_channels = image_channels + num_labels
        self.model = MultiResPatchDiscriminator(num_discriminators,
                                                kernel_size,
                                                num_input_channels,
                                                num_filters,
                                                num_layers,
                                                max_num_filters,
                                                activation_norm_type,
                                                weight_norm_type)
        print('Done with the Multi-resolution patch '
              'discriminator initialization.')

    def forward(self, data, net_G_output, real=True):
        r"""SPADE Generator forward.

        Args:
            data (dict):
              - data  (N x C1 x H x W tensor) : Ground truth images.
              - label (N x C2 x H x W tensor) : Semantic representations.
              - z (N x style_dims tensor): Gaussian random noise.
            net_G_output (dict):
                fake_images  (N x C1 x H x W tensor) : Fake images.
            real (bool): If ``True``, also classifies real images. Otherwise it
                only classifies generated images to save computation during the
                generator update.
        Returns:
            (tuple):
              - real_outputs (list): list of output tensors produced by
              - individual patch discriminators for real images.
              - real_features (list): list of lists of features produced by
                individual patch discriminators for real images.
              - fake_outputs (list): list of output tensors produced by
                individual patch discriminators for fake images.
              - fake_features (list): list of lists of features produced by
                individual patch discriminators for fake images.
        """
        output_x = dict()
        if 'label' in data:
            fake_input_x = torch.cat(
                (data['label'], net_G_output['fake_images']), 1)
        else:
            fake_input_x = net_G_output['fake_images']
        output_x['fake_outputs'], output_x['fake_features'], _ = \
            self.model.forward(fake_input_x)
        if real:
            if 'label' in data:
                real_input_x = torch.cat(
                    (data['label'], data['images']), 1)
            else:
                real_input_x = data['images']
            output_x['real_outputs'], output_x['real_features'], _ = \
                self.model.forward(real_input_x)
        return output_x


class MultiResPatchDiscriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        num_discriminators (int): Num. of discriminators (one per scale).
        kernel_size (int): Convolution kernel size.
        num_image_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self,
                 num_discriminators=3,
                 kernel_size=3,
                 num_image_channels=3,
                 num_filters=64,
                 num_layers=4,
                 max_num_filters=512,
                 activation_norm_type='',
                 weight_norm_type='',
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type' and key != 'patch_wise':
                warnings.warn(
                    "Discriminator argument {} is not used".format(key))

        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            net_discriminator = NLayerPatchDiscriminator(
                kernel_size,
                num_image_channels,
                num_filters,
                num_layers,
                max_num_filters,
                activation_norm_type,
                weight_norm_type)
            self.discriminators.append(net_discriminator)
        print('Done with the Multi-resolution patch '
              'discriminator initialization.')

    def forward(self, input_x):
        r"""Multi-resolution patch discriminator forward.

        Args:
            input_x (tensor) : Input images.
        Returns:
            (tuple):
              - output_list (list): list of output tensors produced by
                individual patch discriminators.
              - features_list (list): list of lists of features produced by
                individual patch discriminators.
              - input_list (list): list of downsampled input images.
        """
        input_list = []
        output_list = []
        features_list = []
        input_downsampled = input_x
        for net_discriminator in self.discriminators:
            input_list.append(input_downsampled)
            output, features = net_discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = nn.functional.interpolate(
                input_downsampled, scale_factor=0.5, mode='bilinear',
                align_corners=True, recompute_scale_factor=True)
        return output_list, features_list, input_list


class WeightSharedMultiResPatchDiscriminator(nn.Module):
    r"""Multi-resolution patch discriminator with shared weights.

    Args:
        num_discriminators (int): Num. of discriminators (one per scale).
        kernel_size (int): Convolution kernel size.
        num_image_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self,
                 num_discriminators=3,
                 kernel_size=3,
                 num_image_channels=3,
                 num_filters=64,
                 num_layers=4,
                 max_num_filters=512,
                 activation_norm_type='',
                 weight_norm_type='',
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type' and key != 'patch_wise':
                warnings.warn(
                    "Discriminator argument {} is not used".format(key))
        self.num_discriminators = num_discriminators
        self.discriminator = NLayerPatchDiscriminator(
            kernel_size,
            num_image_channels,
            num_filters,
            num_layers,
            max_num_filters,
            activation_norm_type,
            weight_norm_type)
        print('Done with the Weight-Shared Multi-resolution patch '
              'discriminator initialization.')

    def forward(self, input_x):
        r"""Multi-resolution patch discriminator forward.

        Args:
            input_x (tensor) : Input images.
        Returns:
            (tuple):
              - output_list (list): list of output tensors produced by
                individual patch discriminators.
              - features_list (list): list of lists of features produced by
                individual patch discriminators.
              - input_list (list): list of downsampled input images.
        """
        input_list = []
        output_list = []
        features_list = []
        input_downsampled = input_x
        for i in range(self.num_discriminators):
            input_list.append(input_downsampled)
            output, features = self.discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = nn.functional.interpolate(
                input_downsampled, scale_factor=0.5, mode='bilinear',
                align_corners=True)
        return output_list, features_list, input_list


class NLayerPatchDiscriminator(nn.Module):
    r"""Patch Discriminator constructor.

    Args:
        kernel_size (int): Convolution kernel size.
        num_input_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self,
                 kernel_size,
                 num_input_channels,
                 num_filters,
                 num_layers,
                 max_num_filters,
                 activation_norm_type,
                 weight_norm_type):
        super(NLayerPatchDiscriminator, self).__init__()
        self.num_layers = num_layers
        padding = int(np.floor((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        base_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity,
                              # inplace_nonlinearity=True,
                              order='CNA')
        layers = [[base_conv2d_block(
            num_input_channels, num_filters, stride=2)]]
        for n in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            stride = 2 if n < (num_layers - 1) else 1
            layers += [[base_conv2d_block(num_filters_prev, num_filters,
                                          stride=stride)]]
        layers += [[Conv2dBlock(num_filters, 1,
                                3, 1,
                                padding,
                                weight_norm_type=weight_norm_type)]]
        for n in range(len(layers)):
            setattr(self, 'layer' + str(n), nn.Sequential(*layers[n]))

    def forward(self, input_x):
        r"""Patch Discriminator forward.

        Args:
            input_x (N x C x H1 x W2 tensor): Concatenation of images and
                semantic representations.
        Returns:
            (tuple):
              - output (N x 1 x H2 x W2 tensor): Discriminator output value.
                Before the sigmoid when using NSGAN.
              - features (list): lists of tensors of the intermediate
                activations.
        """
        res = [input_x]
        for n in range(self.num_layers + 2):
            layer = getattr(self, 'layer' + str(n))
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features
