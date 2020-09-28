# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
import torch.nn as nn

from imaginaire.discriminators.fpse import FPSEDiscriminator
from imaginaire.discriminators.multires_patch import NLayerPatchDiscriminator
from imaginaire.utils.data import (get_paired_input_image_channel_number,
                                   get_paired_input_label_channel_number)
from imaginaire.utils.distributed import master_only_print as print


class Discriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, dis_cfg, data_cfg):
        super(Discriminator, self).__init__()
        print('Multi-resolution patch discriminator initialization.')
        # We assume the first datum is the ground truth image.
        image_channels = get_paired_input_image_channel_number(data_cfg)
        # Calculate number of channels in the input label.
        if data_cfg.type == 'imaginaire.datasets.paired_videos':
            num_labels = get_paired_input_label_channel_number(
                data_cfg, video=True)
        else:
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
        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            net_discriminator = NLayerPatchDiscriminator(
                kernel_size,
                num_input_channels,
                num_filters,
                num_layers,
                max_num_filters,
                activation_norm_type,
                weight_norm_type)
            self.discriminators.append(net_discriminator)
        print('Done with the Multi-resolution patch '
              'discriminator initialization.')
        fpse_kernel_size = getattr(dis_cfg, 'fpse_kernel_size', 3)
        fpse_activation_norm_type = getattr(dis_cfg,
                                            'fpse_activation_norm_type',
                                            'none')
        self.fpse_discriminator = FPSEDiscriminator(
            image_channels,
            num_labels,
            num_filters,
            fpse_kernel_size,
            weight_norm_type,
            fpse_activation_norm_type)

    def _single_forward(self, input_label, input_image):
        # Compute discriminator outputs and intermediate features from input
        # images and semantic labels.
        input_x = torch.cat(
            (input_label, input_image), 1)
        features_list = []
        pred2, pred3, pred4 = self.fpse_discriminator(input_image, input_label)
        output_list = [pred2, pred3, pred4]
        input_downsampled = input_x
        for net_discriminator in self.discriminators:
            output, features = net_discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = nn.functional.interpolate(
                input_downsampled, scale_factor=0.5, mode='bilinear',
                align_corners=True)
        return output_list, features_list

    def forward(self, data, net_G_output):
        r"""SPADE discriminator forward.

        Args:
            data (dict):
              - data  (N x C1 x H x W tensor) : Ground truth images.
              - label (N x C2 x H x W tensor) : Semantic representations.
              - z (N x style_dims tensor): Gaussian random noise.
            net_G_output (dict):
                fake_images  (N x C1 x H x W tensor) : Fake images.
        Returns:
            (dict):
              - real_outputs (list): list of output tensors produced by
                individual patch discriminators for real images.
              - real_features (list): list of lists of features produced by
                individual patch discriminators for real images.
              - fake_outputs (list): list of output tensors produced by
                individual patch discriminators for fake images.
              - fake_features (list): list of lists of features produced by
                individual patch discriminators for fake images.
        """
        output_x = dict()
        output_x['real_outputs'], output_x['real_features'] = \
            self._single_forward(data['label'], data['images'])
        output_x['fake_outputs'], output_x['fake_features'] = \
            self._single_forward(data['label'], net_G_output['fake_images'])
        return output_x
