# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import warnings

import torch
from torch import nn

from imaginaire.layers import Conv2dBlock, Res2dBlock


class Discriminator(nn.Module):
    r"""Discriminator in the improved FUNIT baseline in the COCO-FUNIT paper.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, dis_cfg, data_cfg):
        super().__init__()
        self.model = ResDiscriminator(**vars(dis_cfg))

    def forward(self, data, net_G_output, recon=True):
        r"""Improved FUNIT discriminator forward function.

        Args:
            data (dict): Training data at the current iteration.
            net_G_output (dict): Fake data generated at the current iteration.
            recon (bool): If ``True``, also classifies reconstructed images.
        """
        source_labels = data['labels_content']
        target_labels = data['labels_style']
        fake_out_trans, fake_features_trans = \
            self.model(net_G_output['images_trans'], target_labels)
        output = dict(fake_out_trans=fake_out_trans,
                      fake_features_trans=fake_features_trans)

        real_out_style, real_features_style = \
            self.model(data['images_style'], target_labels)
        output.update(dict(real_out_style=real_out_style,
                           real_features_style=real_features_style))
        if recon:
            fake_out_recon, fake_features_recon = \
                self.model(net_G_output['images_recon'], source_labels)
            output.update(dict(fake_out_recon=fake_out_recon,
                               fake_features_recon=fake_features_recon))
        return output


class ResDiscriminator(nn.Module):
    r"""Residual discriminator architecture used in the FUNIT paper."""

    def __init__(self,
                 image_channels=3,
                 num_classes=119,
                 num_filters=64,
                 max_num_filters=1024,
                 num_layers=6,
                 padding_mode='reflect',
                 weight_norm_type='',
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type':
                warnings.warn(
                    "Discriminator argument {} is not used".format(key))

        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type='none',
                           weight_norm_type=weight_norm_type,
                           bias=[True, True, True],
                           nonlinearity='leakyrelu',
                           order='NACNAC')

        first_kernel_size = 7
        first_padding = (first_kernel_size - 1) // 2
        model = [Conv2dBlock(image_channels, num_filters,
                             first_kernel_size, 1, first_padding,
                             padding_mode=padding_mode,
                             weight_norm_type=weight_norm_type)]
        for i in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            model += [Res2dBlock(num_filters_prev, num_filters_prev,
                                 **conv_params),
                      Res2dBlock(num_filters_prev, num_filters,
                                 **conv_params)]
            if i != num_layers - 1:
                model += [nn.ReflectionPad2d(1),
                          nn.AvgPool2d(3, stride=2)]
        self.model = nn.Sequential(*model)
        self.classifier = Conv2dBlock(num_filters, 1, 1, 1, 0,
                                      nonlinearity='leakyrelu',
                                      weight_norm_type=weight_norm_type,
                                      order='NACNAC')

        self.embedder = nn.Embedding(num_classes, num_filters)

    def forward(self, images, labels=None):
        r"""Forward function of the projection discriminator.

        Args:
            images (image tensor): Images inputted to the discriminator.
            labels (long int tensor): Class labels of the images.
        """
        assert (images.size(0) == labels.size(0))
        features = self.model(images)
        outputs = self.classifier(features)
        features_1x1 = features.mean(3).mean(2)
        if labels is None:
            return features_1x1
        embeddings = self.embedder(labels)
        outputs += torch.sum(embeddings * features_1x1, dim=1,
                             keepdim=True).view(images.size(0), 1, 1, 1)
        return outputs, features_1x1
