# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from imaginaire.layers import Conv2dBlock

from imaginaire.utils.data import get_paired_input_label_channel_number, get_paired_input_image_channel_number
from imaginaire.utils.distributed import master_only_print as print


class Discriminator(nn.Module):
    r"""Multi-resolution patch discriminator. Based on FPSE discriminator but with N+1 labels.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, dis_cfg, data_cfg):
        super(Discriminator, self).__init__()
        # We assume the first datum is the ground truth image.
        image_channels = get_paired_input_image_channel_number(data_cfg)
        # Calculate number of channels in the input label.
        num_labels = get_paired_input_label_channel_number(data_cfg)

        self.use_label = getattr(dis_cfg, 'use_label', True)
        # Override number of input channels
        if hasattr(dis_cfg, 'image_channels'):
            image_channels = dis_cfg.image_channels
        if hasattr(dis_cfg, 'num_labels'):
            num_labels = dis_cfg.num_labels
        else:
            # We assume the first datum is the ground truth image.
            image_channels = get_paired_input_image_channel_number(data_cfg)
            # Calculate number of channels in the input label.
            num_labels = get_paired_input_label_channel_number(data_cfg)

        if not self.use_label:
            num_labels = 2  # ignore + true

        # Build the discriminator.
        num_filters = getattr(dis_cfg, 'num_filters', 128)
        weight_norm_type = getattr(dis_cfg, 'weight_norm_type', 'spectral')

        fpse_kernel_size = getattr(dis_cfg, 'fpse_kernel_size', 3)
        fpse_activation_norm_type = getattr(dis_cfg,
                                            'fpse_activation_norm_type',
                                            'none')
        do_multiscale = getattr(dis_cfg, 'do_multiscale', False)
        smooth_resample = getattr(dis_cfg, 'smooth_resample', False)
        no_label_except_largest_scale = getattr(dis_cfg, 'no_label_except_largest_scale', False)

        self.fpse_discriminator = FPSEDiscriminator(
            image_channels,
            num_labels,
            num_filters,
            fpse_kernel_size,
            weight_norm_type,
            fpse_activation_norm_type,
            do_multiscale,
            smooth_resample,
            no_label_except_largest_scale)

    def _single_forward(self, input_label, input_image, weights):
        output_list, features_list = self.fpse_discriminator(input_image, input_label, weights)
        return output_list, [features_list]

    def forward(self, data, net_G_output, weights=None, incl_real=False, incl_pseudo_real=False):
        r"""GANcraft discriminator forward.

        Args:
            data (dict):
              - data  (N x C1 x H x W tensor) : Ground truth images.
              - label (N x C2 x H x W tensor) : Semantic representations.
              - z (N x style_dims tensor): Gaussian random noise.
            net_G_output (dict):
              - fake_images  (N x C1 x H x W tensor) : Fake images.
        Returns:
            output_x (dict):
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

        # Fake.
        fake_images = net_G_output['fake_images']
        if self.use_label:
            fake_labels = data['fake_masks']
        else:
            fake_labels = torch.zeros([fake_images.size(0), 2, fake_images.size(
                2), fake_images.size(3)], device=fake_images.device, dtype=fake_images.dtype)
            fake_labels[:, 1, :, :] = 1
        output_x['fake_outputs'], output_x['fake_features'] = \
            self._single_forward(fake_labels, fake_images, None)

        # Real.
        if incl_real:
            real_images = data['images']
            if self.use_label:
                real_labels = data['real_masks']
            else:
                real_labels = torch.zeros([real_images.size(0), 2, real_images.size(
                    2), real_images.size(3)], device=real_images.device, dtype=real_images.dtype)
                real_labels[:, 1, :, :] = 1
            output_x['real_outputs'], output_x['real_features'] = \
                self._single_forward(real_labels, real_images, None)

        # pseudo-Real.
        if incl_pseudo_real:
            preal_images = data['pseudo_real_img']
            preal_labels = data['fake_masks']
            if not self.use_label:
                preal_labels = torch.zeros([preal_images.size(0), 2, preal_images.size(
                    2), preal_images.size(3)], device=preal_images.device, dtype=preal_images.dtype)
                preal_labels[:, 1, :, :] = 1
            output_x['pseudo_real_outputs'], output_x['pseudo_real_features'] = \
                self._single_forward(preal_labels, preal_images, None)

        return output_x


class FPSEDiscriminator(nn.Module):
    def __init__(self,
                 num_input_channels,
                 num_labels,
                 num_filters,
                 kernel_size,
                 weight_norm_type,
                 activation_norm_type,
                 do_multiscale,
                 smooth_resample,
                 no_label_except_largest_scale):
        super().__init__()

        self.do_multiscale = do_multiscale
        self.no_label_except_largest_scale = no_label_except_largest_scale

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
        self.enc1 = down_conv2d_block(num_input_channels, num_filters)  # 3
        self.enc2 = down_conv2d_block(1 * num_filters, 2 * num_filters)  # 7
        self.enc3 = down_conv2d_block(2 * num_filters, 4 * num_filters)  # 15
        self.enc4 = down_conv2d_block(4 * num_filters, 8 * num_filters)  # 31
        self.enc5 = down_conv2d_block(8 * num_filters, 8 * num_filters)  # 63

        # top-down pathway
        # self.lat1 = latent_conv2d_block(num_filters, 2 * num_filters) # Zekun
        self.lat2 = latent_conv2d_block(2 * num_filters, 4 * num_filters)
        self.lat3 = latent_conv2d_block(4 * num_filters, 4 * num_filters)
        self.lat4 = latent_conv2d_block(8 * num_filters, 4 * num_filters)
        self.lat5 = latent_conv2d_block(8 * num_filters, 4 * num_filters)

        # upsampling
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=False)

        # final layers
        self.final2 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
        self.output = Conv2dBlock(num_filters * 2, num_labels+1, kernel_size=1)

        if self.do_multiscale:
            self.final3 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
            self.final4 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
            if self.no_label_except_largest_scale:
                self.output3 = Conv2dBlock(num_filters * 2, 2, kernel_size=1)
                self.output4 = Conv2dBlock(num_filters * 2, 2, kernel_size=1)
            else:
                self.output3 = Conv2dBlock(num_filters * 2, num_labels+1, kernel_size=1)
                self.output4 = Conv2dBlock(num_filters * 2, num_labels+1, kernel_size=1)

        self.interpolator = functools.partial(F.interpolate, mode='nearest')
        if smooth_resample:
            self.interpolator = self.smooth_interp

    @staticmethod
    def smooth_interp(x, size):
        r"""Smooth interpolation of segmentation maps.

        Args:
            x (4D tensor): Segmentation maps.
            size(2D list): Target size (H, W).
        """
        x = F.interpolate(x, size=size, mode='area')
        onehot_idx = torch.argmax(x, dim=-3, keepdims=True)
        x.fill_(0.0)
        x.scatter_(1, onehot_idx, 1.0)
        return x

    # Weights: [N C]
    def forward(self, images, segmaps, weights=None):
        # Assume images 256x256
        # bottom-up pathway
        feat11 = self.enc1(images)  # 128
        feat12 = self.enc2(feat11)  # 64
        feat13 = self.enc3(feat12)  # 32
        feat14 = self.enc4(feat13)  # 16
        feat15 = self.enc5(feat14)  # 8
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)  # 8
        feat24 = self.upsample2x(feat25) + self.lat4(feat14)  # 16
        feat23 = self.upsample2x(feat24) + self.lat3(feat13)  # 32
        feat22 = self.upsample2x(feat23) + self.lat2(feat12)  # 64

        # final prediction layers
        feat32 = self.final2(feat22)

        results = []
        label_map = self.interpolator(segmaps, size=feat32.size()[2:])
        pred2 = self.output(feat32)  # N, num_labels+1, H//4, W//4

        features = [feat11, feat12, feat13, feat14, feat15, feat25, feat24, feat23, feat22]
        if weights is not None:
            label_map = label_map * weights[..., None, None]
        results.append({'pred': pred2, 'label': label_map})

        if self.do_multiscale:
            feat33 = self.final3(feat23)
            pred3 = self.output3(feat33)

            feat34 = self.final4(feat24)
            pred4 = self.output4(feat34)

            if self.no_label_except_largest_scale:
                label_map3 = torch.ones([pred3.size(0), 1, pred3.size(2), pred3.size(3)], device=pred3.device)
                label_map4 = torch.ones([pred4.size(0), 1, pred4.size(2), pred4.size(3)], device=pred4.device)
            else:
                label_map3 = self.interpolator(segmaps, size=pred3.size()[2:])
                label_map4 = self.interpolator(segmaps, size=pred4.size()[2:])

            if weights is not None:
                label_map3 = label_map3 * weights[..., None, None]
                label_map4 = label_map4 * weights[..., None, None]

            results.append({'pred': pred3, 'label': label_map3})
            results.append({'pred': pred4, 'label': label_map4})

        return results, features
