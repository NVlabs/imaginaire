# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools
import math
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Upsample as NearestUpsample

from imaginaire.layers import Conv2dBlock, LinearBlock, Res2dBlock
from imaginaire.utils.data import (get_crop_h_w,
                                   get_paired_input_image_channel_number,
                                   get_paired_input_label_channel_number)
from imaginaire.utils.distributed import master_only_print as print


class Generator(nn.Module):
    r"""SPADE generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super(Generator, self).__init__()
        print('SPADE generator initialization.')
        # We assume the first datum is the ground truth image.
        image_channels = get_paired_input_image_channel_number(data_cfg)
        # Calculate number of channels in the input label.
        num_labels = get_paired_input_label_channel_number(data_cfg)
        crop_h, crop_w = get_crop_h_w(data_cfg.train.augmentations)
        # Build the generator
        out_image_small_side_size = crop_w if crop_w < crop_h else crop_h
        num_filters = getattr(gen_cfg, 'num_filters', 128)
        kernel_size = getattr(gen_cfg, 'kernel_size', 3)
        weight_norm_type = getattr(gen_cfg, 'weight_norm_type', 'spectral')

        cond_dims = 0
        # Check whether we use the style code.
        style_dims = getattr(gen_cfg, 'style_dims', None)
        self.style_dims = style_dims
        if style_dims is not None:
            print('\tStyle code dimensions: %d' % style_dims)
            cond_dims += style_dims
            self.use_style = True
        else:
            self.use_style = False
        # Check whether we use the attribute code.
        if hasattr(gen_cfg, 'attribute_dims'):
            self.use_attribute = True
            self.attribute_dims = gen_cfg.attribute_dims
            cond_dims += gen_cfg.attribute_dims
        else:
            self.use_attribute = False

        if not self.use_style and not self.use_attribute:
            self.use_style_encoder = False
        else:
            self.use_style_encoder = True
        print('\tBase filter number: %d' % num_filters)
        print('\tConvolution kernel size: %d' % kernel_size)
        print('\tWeight norm type: %s' % weight_norm_type)
        skip_activation_norm = \
            getattr(gen_cfg, 'skip_activation_norm', True)
        activation_norm_params = \
            getattr(gen_cfg, 'activation_norm_params', None)
        if activation_norm_params is None:
            activation_norm_params = types.SimpleNamespace()
        if not hasattr(activation_norm_params, 'num_filters'):
            setattr(activation_norm_params, 'num_filters', 128)
        if not hasattr(activation_norm_params, 'kernel_size'):
            setattr(activation_norm_params, 'kernel_size', 3)
        if not hasattr(activation_norm_params, 'activation_norm_type'):
            setattr(activation_norm_params,
                    'activation_norm_type', 'sync_batch')
        if not hasattr(activation_norm_params, 'separate_projection'):
            setattr(activation_norm_params, 'separate_projection', False)
        if not hasattr(activation_norm_params, 'activation_norm_params'):
            activation_norm_params.activation_norm_params = \
                types.SimpleNamespace()
            activation_norm_params.activation_norm_params.affine = True
        setattr(activation_norm_params, 'cond_dims', num_labels)
        if not hasattr(activation_norm_params, 'weight_norm_type'):
            setattr(activation_norm_params,
                    'weight_norm_type', weight_norm_type)
        global_adaptive_norm_type = getattr(gen_cfg,
                                            'global_adaptive_norm_type',
                                            'sync_batch')
        use_posenc_in_input_layer = getattr(gen_cfg,
                                            'use_posenc_in_input_layer',
                                            True)
        print(activation_norm_params)
        self.spade_generator = SPADEGenerator(num_labels,
                                              out_image_small_side_size,
                                              image_channels,
                                              num_filters,
                                              kernel_size,
                                              cond_dims,
                                              activation_norm_params,
                                              weight_norm_type,
                                              global_adaptive_norm_type,
                                              skip_activation_norm,
                                              use_posenc_in_input_layer,
                                              self.use_style_encoder)
        if self.use_style:
            # Build the encoder.
            style_enc_cfg = getattr(gen_cfg, 'style_enc', None)
            if style_enc_cfg is None:
                style_enc_cfg = types.SimpleNamespace()
            if not hasattr(style_enc_cfg, 'num_filters'):
                setattr(style_enc_cfg, 'num_filters', 128)
            if not hasattr(style_enc_cfg, 'kernel_size'):
                setattr(style_enc_cfg, 'kernel_size', 3)
            if not hasattr(style_enc_cfg, 'weight_norm_type'):
                setattr(style_enc_cfg, 'weight_norm_type', weight_norm_type)
            setattr(style_enc_cfg, 'input_image_channels', image_channels)
            setattr(style_enc_cfg, 'style_dims', style_dims)
            self.style_encoder = StyleEncoder(style_enc_cfg)

        self.z = None
        print('Done with the SPADE generator initialization.')

    def forward(self, data, random_style=False):
        r"""SPADE Generator forward.

        Args:
            data (dict):
              - images (N x C1 x H x W tensor) : Ground truth images
              - label (N x C2 x H x W tensor) : Semantic representations
              - z (N x style_dims tensor): Gaussian random noise
              - random_style (bool): Whether to sample a random style vector.
        Returns:
            (dict):
              - fake_images (N x 3 x H x W tensor): fake images
              - mu (N x C1 tensor): mean vectors
              - logvar (N x C1 tensor): log-variance vectors
        """
        if self.use_style_encoder:
            if random_style:
                bs = data['label'].size(0)
                z = torch.randn(
                    bs, self.style_dims, dtype=torch.float32).cuda()
                if (data['label'].dtype ==
                        data['label'].dtype == torch.float16):
                    z = z.half()
                mu = None
                logvar = None
            else:
                mu, logvar, z = self.style_encoder(data['images'])
            if self.use_attribute:
                data['z'] = torch.cat((z, data['attributes'].squeeze(1)), dim=1)
            else:
                data['z'] = z
        output = self.spade_generator(data)
        if self.use_style_encoder:
            output['mu'] = mu
            output['logvar'] = logvar
        return output

    def inference(self,
                  data,
                  random_style=False,
                  use_fixed_random_style=False,
                  keep_original_size=False):
        r"""Compute results images for a batch of input data and save the
        results in the specified folder.

        Args:
            data (dict):
              - images (N x C1 x H x W tensor) : Ground truth images
              - label (N x C2 x H x W tensor) : Semantic representations
              - z (N x style_dims tensor): Gaussian random noise
            random_style (bool): Whether to sample a random style vector.
            use_fixed_random_style (bool): Sample random style once and use it
                for all the remaining inference.
            keep_original_size (bool): Keep original size of the input.
        Returns:
            (dict):
              - fake_images (N x 3 x H x W tensor): fake images
              - mu (N x C1 tensor): mean vectors
              - logvar (N x C1 tensor): log-variance vectors
        """
        self.eval()
        self.spade_generator.eval()
        if random_style:
            if self.z is None or not use_fixed_random_style:
                bs = data['label'].size(0)
                z = torch.randn(
                    bs, self.style_dims, dtype=torch.float32).to('cuda')
                if data['label'].dtype == data['label'].dtype == torch.float16:
                    z = z.half()
                self.z = z
            else:
                z = self.z
        else:
            mu, logvar, z = self.style_encoder(data['images'])
        data['z'] = z
        output = self.spade_generator(data)
        output_images = output['fake_images']
        if keep_original_size:
            height = data['original_h_w'][0][0]
            width = data['original_h_w'][0][1]
            output_images = torch.nn.functional.interpolate(
                output_images, size=[height, width])
        file_names = data['key']['seg_maps'][0]
        return output_images, file_names


class SPADEGenerator(nn.Module):
    r"""SPADE Image Generator constructor.

    Args:
        num_labels (int): Number of different labels.
        out_image_small_side_size (int): min(width, height)
        image_channels (int): Num. of channels of the output image.
        num_filters (int): Base filter numbers.
        kernel_size (int): Convolution kernel size.
        style_dims (int): Dimensions of the style code.
        activation_norm_params (obj): Spatially adaptive normalization param.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        global_adaptive_norm_type (str): Type of normalization in SPADE.
        skip_activation_norm (bool): If ``True``, applies activation norm to the
            shortcut connection in residual blocks.
        use_style_encoder (bool): Whether to use global adaptive norm
            like conditional batch norm or adaptive instance norm.
    """

    def __init__(self,
                 num_labels,
                 out_image_small_side_size,
                 image_channels,
                 num_filters,
                 kernel_size,
                 style_dims,
                 activation_norm_params,
                 weight_norm_type,
                 global_adaptive_norm_type,
                 skip_activation_norm,
                 use_posenc_in_input_layer,
                 use_style_encoder):
        super(SPADEGenerator, self).__init__()
        self.use_style_encoder = use_style_encoder
        self.use_posenc_in_input_layer = use_posenc_in_input_layer
        self.out_image_small_side_size = out_image_small_side_size
        self.num_filters = num_filters
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        activation_norm_type = 'spatially_adaptive'
        base_res2d_block = \
            functools.partial(Res2dBlock,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=[True, True, False],
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              activation_norm_params=activation_norm_params,
                              skip_activation_norm=skip_activation_norm,
                              nonlinearity=nonlinearity,
                              order='NACNAC')
        if self.use_style_encoder:
            self.fc_0 = LinearBlock(style_dims, 2 * style_dims,
                                    weight_norm_type=weight_norm_type,
                                    nonlinearity='relu',
                                    order='CAN')
            self.fc_1 = LinearBlock(2 * style_dims, 2 * style_dims,
                                    weight_norm_type=weight_norm_type,
                                    nonlinearity='relu',
                                    order='CAN')

            adaptive_norm_params = types.SimpleNamespace()
            if not hasattr(adaptive_norm_params, 'cond_dims'):
                setattr(adaptive_norm_params, 'cond_dims', 2 * style_dims)
            if not hasattr(adaptive_norm_params, 'activation_norm_type'):
                setattr(adaptive_norm_params, 'activation_norm_type',
                        global_adaptive_norm_type)
            if not hasattr(adaptive_norm_params, 'weight_norm_type'):
                setattr(adaptive_norm_params,
                        'weight_norm_type',
                        activation_norm_params.weight_norm_type)
            if not hasattr(adaptive_norm_params, 'separate_projection'):
                setattr(adaptive_norm_params, 'separate_projection',
                        activation_norm_params.separate_projection)
            adaptive_norm_params.activation_norm_params = \
                types.SimpleNamespace()
            setattr(adaptive_norm_params.activation_norm_params, 'affine',
                    activation_norm_params.activation_norm_params.affine)
            base_cbn2d_block = \
                functools.partial(Conv2dBlock,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=padding,
                                  bias=True,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type='adaptive',
                                  activation_norm_params=adaptive_norm_params,
                                  nonlinearity=nonlinearity,
                                  order='NAC')
        else:
            base_conv2d_block = \
                functools.partial(Conv2dBlock,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=padding,
                                  bias=True,
                                  weight_norm_type=weight_norm_type,
                                  nonlinearity=nonlinearity,
                                  order='NAC')
        in_num_labels = num_labels
        in_num_labels += 2 if self.use_posenc_in_input_layer else 0
        self.head_0 = Conv2dBlock(in_num_labels, 8 * num_filters,
                                  kernel_size=kernel_size, stride=1,
                                  padding=padding,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type='none',
                                  nonlinearity=nonlinearity)
        if self.use_style_encoder:
            self.cbn_head_0 = base_cbn2d_block(
                8 * num_filters, 16 * num_filters)
        else:
            self.conv_head_0 = base_conv2d_block(
                8 * num_filters, 16 * num_filters)
        self.head_1 = base_res2d_block(16 * num_filters, 16 * num_filters)
        self.head_2 = base_res2d_block(16 * num_filters, 16 * num_filters)

        self.up_0a = base_res2d_block(16 * num_filters, 8 * num_filters)
        if self.use_style_encoder:
            self.cbn_up_0a = base_cbn2d_block(
                8 * num_filters, 8 * num_filters)
        else:
            self.conv_up_0a = base_conv2d_block(
                8 * num_filters, 8 * num_filters)
        self.up_0b = base_res2d_block(8 * num_filters, 8 * num_filters)

        self.up_1a = base_res2d_block(8 * num_filters, 4 * num_filters)
        if self.use_style_encoder:
            self.cbn_up_1a = base_cbn2d_block(
                4 * num_filters, 4 * num_filters)
        else:
            self.conv_up_1a = base_conv2d_block(
                4 * num_filters, 4 * num_filters)
        self.up_1b = base_res2d_block(4 * num_filters, 4 * num_filters)
        self.up_2a = base_res2d_block(4 * num_filters, 4 * num_filters)
        if self.use_style_encoder:
            self.cbn_up_2a = base_cbn2d_block(
                4 * num_filters, 4 * num_filters)
        else:
            self.conv_up_2a = base_conv2d_block(
                4 * num_filters, 4 * num_filters)
        self.up_2b = base_res2d_block(4 * num_filters, 2 * num_filters)
        self.conv_img256 = Conv2dBlock(2 * num_filters, image_channels,
                                       5, stride=1, padding=2,
                                       weight_norm_type=weight_norm_type,
                                       activation_norm_type='none',
                                       nonlinearity=nonlinearity,
                                       order='ANC')
        self.base = 16
        if self.out_image_small_side_size == 512:
            self.up_3a = base_res2d_block(2 * num_filters, 1 * num_filters)
            self.up_3b = base_res2d_block(1 * num_filters, 1 * num_filters)
            self.conv_img512 = Conv2dBlock(1 * num_filters, image_channels,
                                           5, stride=1, padding=2,
                                           weight_norm_type=weight_norm_type,
                                           activation_norm_type='none',
                                           nonlinearity=nonlinearity,
                                           order='ANC')
            self.base = 32
        if self.out_image_small_side_size == 1024:
            self.up_3a = base_res2d_block(2 * num_filters, 1 * num_filters)
            self.up_3b = base_res2d_block(1 * num_filters, 1 * num_filters)
            self.up_4a = base_res2d_block(num_filters, num_filters // 2)
            self.up_4b = base_res2d_block(num_filters // 2, num_filters // 2)
            self.conv_img1024 = Conv2dBlock(num_filters // 2, image_channels,
                                            5, stride=1, padding=2,
                                            weight_norm_type=weight_norm_type,
                                            activation_norm_type='none',
                                            nonlinearity=nonlinearity,
                                            order='ANC')
            self.base = 64
        if self.out_image_small_side_size != 256 and \
                self.out_image_small_side_size != \
                512 and self.out_image_small_side_size != 1024:
            raise ValueError('Generation image size (%d, %d) not supported' %
                             (self.out_image_small_side_size,
                              self.out_image_small_side_size))
        self.nearest_upsample2x = NearestUpsample(scale_factor=2,
                                                  mode='nearest')
        xv, yv = torch.meshgrid(
            [torch.arange(-1, 1.1, 2. / 15), torch.arange(-1, 1.1, 2. / 15)])
        self.xy = torch.cat((xv.unsqueeze(0), yv.unsqueeze(0)), 0).unsqueeze(0)
        self.xy = self.xy.cuda()

    def forward(self, data):
        r"""SPADE Generator forward.

        Args:
            data (dict):
              - data  (N x C1 x H x W tensor) : Ground truth images.
              - label (N x C2 x H x W tensor) : Semantic representations.
              - z (N x style_dims tensor): Gaussian random noise.
        Returns:
            output (dict):
              - fake_images (N x 3 x H x W tensor): Fake images.
        """
        seg = data['label']

        if self.use_style_encoder:
            z = data['z']
            z = self.fc_0(z)
            z = self.fc_1(z)

        # The code piece below makes sure that the input size is always 16x16
        sy = math.floor(seg.size()[2] * 1.0 / self.base)
        sx = math.floor(seg.size()[3] * 1.0 / self.base)

        in_seg = F.interpolate(seg, size=[sy, sx], mode='nearest')
        if self.use_posenc_in_input_layer:
            in_xy = F.interpolate(self.xy, size=[sy, sx], mode='bicubic')
            in_seg_xy = torch.cat(
                (in_seg, in_xy.expand(in_seg.size()[0], 2, sy, sx)), 1)
        else:
            in_seg_xy = in_seg
        # 16x16
        x = self.head_0(in_seg_xy)
        if self.use_style_encoder:
            x = self.cbn_head_0(x, z)
        else:
            x = self.conv_head_0(x)
        x = self.head_1(x, seg)
        x = self.head_2(x, seg)
        x = self.nearest_upsample2x(x)
        # 32x32
        x = self.up_0a(x, seg)
        if self.use_style_encoder:
            x = self.cbn_up_0a(x, z)
        else:
            x = self.conv_up_0a(x)
        x = self.up_0b(x, seg)
        x = self.nearest_upsample2x(x)
        # 64x64
        x = self.up_1a(x, seg)
        if self.use_style_encoder:
            x = self.cbn_up_1a(x, z)
        else:
            x = self.conv_up_1a(x)
        x = self.up_1b(x, seg)
        x = self.nearest_upsample2x(x)
        # 128x128
        x = self.up_2a(x, seg)
        if self.use_style_encoder:
            x = self.cbn_up_2a(x, z)
        else:
            x = self.conv_up_2a(x)
        x = self.up_2b(x, seg)
        x = self.nearest_upsample2x(x)
        # 256x256
        if self.out_image_small_side_size == 256:
            x256 = self.conv_img256(x)
            x = torch.tanh(x256)
        # 512x512
        elif self.out_image_small_side_size == 512:
            x256 = self.conv_img256(x)
            x256 = self.nearest_upsample2x(x256)
            x = self.up_3a(x, seg)
            x = self.up_3b(x, seg)
            x = self.nearest_upsample2x(x)
            x512 = self.conv_img512(x)
            x = torch.tanh(x256 + x512)
        # 1024x1024
        elif self.out_image_small_side_size == 1024:
            x256 = self.conv_img256(x)
            x256 = self.nearest_upsample2x(x256)
            x = self.up_3a(x, seg)
            x = self.up_3b(x, seg)
            x = self.nearest_upsample2x(x)
            x512 = self.conv_img512(x)
            x512 = self.nearest_upsample2x(x512)
            x = self.up_4a(x, seg)
            x = self.up_4b(x, seg)
            x = self.nearest_upsample2x(x)
            x1024 = self.conv_img1024(x)
            x = torch.tanh(x256 + x512 + x1024)
        output = dict()
        output['fake_images'] = x
        return output


class StyleEncoder(nn.Module):
    r"""Style Encode constructor.

    Args:
        style_enc_cfg (obj): Style encoder definition file.
    """

    def __init__(self, style_enc_cfg):
        super(StyleEncoder, self).__init__()
        input_image_channels = style_enc_cfg.input_image_channels
        num_filters = style_enc_cfg.num_filters
        kernel_size = style_enc_cfg.kernel_size
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        style_dims = style_enc_cfg.style_dims
        weight_norm_type = style_enc_cfg.weight_norm_type
        activation_norm_type = 'none'
        nonlinearity = 'leakyrelu'
        base_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              # inplace_nonlinearity=True,
                              nonlinearity=nonlinearity)
        self.layer1 = base_conv2d_block(input_image_channels, num_filters)
        self.layer2 = base_conv2d_block(num_filters * 1, num_filters * 2)
        self.layer3 = base_conv2d_block(num_filters * 2, num_filters * 4)
        self.layer4 = base_conv2d_block(num_filters * 4, num_filters * 8)
        self.layer5 = base_conv2d_block(num_filters * 8, num_filters * 8)
        self.layer6 = base_conv2d_block(num_filters * 8, num_filters * 8)
        self.fc_mu = LinearBlock(num_filters * 8 * 4 * 4, style_dims)
        self.fc_var = LinearBlock(num_filters * 8 * 4 * 4, style_dims)

    def forward(self, input_x):
        r"""SPADE Style Encoder forward.

        Args:
            input_x (N x 3 x H x W tensor): input images.
        Returns:
            (tuple):
              - mu (N x C tensor): Mean vectors.
              - logvar (N x C tensor): Log-variance vectors.
              - z (N x C tensor): Style code vectors.
        """
        if input_x.size(2) != 256 or input_x.size(3) != 256:
            input_x = F.interpolate(input_x, size=(256, 256), mode='bilinear')
        x = self.layer1(input_x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std) + mu
        return mu, logvar, z
