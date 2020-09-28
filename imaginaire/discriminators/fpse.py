# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imaginaire.layers import Conv2dBlock


class FPSEDiscriminator(nn.Module):
    r"""# Feature-Pyramid Semantics Embedding Discriminator. This is a copy
    of the discriminator in https://arxiv.org/pdf/1910.06809.pdf
    """

    def __init__(self,
                 num_input_channels,
                 num_labels,
                 num_filters,
                 kernel_size,
                 weight_norm_type,
                 activation_norm_type):
        super().__init__()
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        stride1_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity,
                              # inplace_nonlinearity=True,
                              order='CNA')
        down_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity,
                              # inplace_nonlinearity=True,
                              order='CNA')
        latent_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=1,
                              stride=1,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity,
                              # inplace_nonlinearity=True,
                              order='CNA')
        # bottom-up pathway

        self.enc1 = down_conv2d_block(num_input_channels, num_filters)
        self.enc2 = down_conv2d_block(1 * num_filters, 2 * num_filters)
        self.enc3 = down_conv2d_block(2 * num_filters, 4 * num_filters)
        self.enc4 = down_conv2d_block(4 * num_filters, 8 * num_filters)
        self.enc5 = down_conv2d_block(8 * num_filters, 8 * num_filters)

        # top-down pathway
        self.lat2 = latent_conv2d_block(2 * num_filters, 4 * num_filters)
        self.lat3 = latent_conv2d_block(4 * num_filters, 4 * num_filters)
        self.lat4 = latent_conv2d_block(8 * num_filters, 4 * num_filters)
        self.lat5 = latent_conv2d_block(8 * num_filters, 4 * num_filters)

        # upsampling
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=False)

        # final layers
        self.final2 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
        self.final3 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
        self.final4 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)

        # true/false prediction and semantic alignment prediction
        self.output = Conv2dBlock(num_filters * 2, 1, kernel_size=1)
        self.seg = Conv2dBlock(num_filters * 2, num_filters * 2, kernel_size=1)
        self.embedding = Conv2dBlock(num_labels, num_filters * 2, kernel_size=1)

    def forward(self, images, segmaps):
        r"""

        Args:
            images: image tensors.
            segmaps: segmentation map tensors.
        """
        # bottom-up pathway
        feat11 = self.enc1(images)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.upsample2x(feat25) + self.lat4(feat14)
        feat23 = self.upsample2x(feat24) + self.lat3(feat13)
        feat22 = self.upsample2x(feat23) + self.lat2(feat12)
        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)
        # Patch-based True/False prediction
        pred2 = self.output(feat32)
        pred3 = self.output(feat33)
        pred4 = self.output(feat34)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # # segmentation map embedding
        segembs = self.embedding(segmaps)
        segembs = F.avg_pool2d(segembs, kernel_size=2, stride=2)
        segembs2 = F.avg_pool2d(segembs, kernel_size=2, stride=2)
        segembs3 = F.avg_pool2d(segembs2, kernel_size=2, stride=2)
        segembs4 = F.avg_pool2d(segembs3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += torch.mul(segembs2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segembs3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segembs4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        # results = [pred2, pred3, pred4]

        return pred2, pred3, pred4
