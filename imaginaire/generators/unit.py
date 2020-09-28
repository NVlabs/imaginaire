# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import warnings

from torch import nn
from torch.nn import Upsample as NearestUpsample

from imaginaire.layers import Conv2dBlock, Res2dBlock


class Generator(nn.Module):
    r"""Improved UNIT generator.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.autoencoder_a = AutoEncoder(**vars(gen_cfg))
        self.autoencoder_b = AutoEncoder(**vars(gen_cfg))

    def forward(self, data, image_recon=True, cycle_recon=True):
        r"""UNIT forward function"""
        images_a = data['images_a']
        images_b = data['images_b']
        net_G_output = dict()

        # encode input images into latent code
        content_a = self.autoencoder_a.content_encoder(images_a)
        content_b = self.autoencoder_b.content_encoder(images_b)

        # decode (within domain)
        if image_recon:
            images_aa = self.autoencoder_a.decoder(content_a)
            images_bb = self.autoencoder_b.decoder(content_b)
            net_G_output.update(dict(images_aa=images_aa, images_bb=images_bb))

        # decode (cross domain)
        images_ba = self.autoencoder_a.decoder(content_b)
        images_ab = self.autoencoder_b.decoder(content_a)

        # cycle reconstruction
        if cycle_recon:
            content_ba = self.autoencoder_a.content_encoder(images_ba)
            content_ab = self.autoencoder_b.content_encoder(images_ab)
            images_aba = self.autoencoder_a.decoder(content_ab)
            images_bab = self.autoencoder_b.decoder(content_ba)
            net_G_output.update(
                dict(content_ba=content_ba, content_ab=content_ab,
                     images_aba=images_aba, images_bab=images_bab))

        # required outputs
        net_G_output.update(dict(content_a=content_a, content_b=content_b,
                                 images_ba=images_ba, images_ab=images_ab))

        return net_G_output

    def inference(self, data, a2b=True):
        r"""UNIT inference.

        Args:
            data (dict): Training data at the current iteration.
              - images_a (tensor): Images from domain A.
              - images_b (tensor): Images from domain B.
            a2b (bool): If ``True``, translates images from domain A to B,
                otherwise from B to A.
        """
        if a2b:
            input_key = 'images_a'
            content_encode = self.autoencoder_a.content_encoder
            decode = self.autoencoder_b.decoder
        else:
            input_key = 'images_b'
            content_encode = self.autoencoder_b.content_encoder
            decode = self.autoencoder_a.decoder

        content_images = data[input_key]
        content = content_encode(content_images)
        output_images = decode(content)
        filename = '%s/%s' % (
            data['key'][input_key]['sequence_name'][0],
            data['key'][input_key]['filename'][0])
        filenames = [filename]
        return output_images, filenames


class AutoEncoder(nn.Module):
    r"""Improved UNIT autoencoder.

    Args:
        num_filters (int): Base filter numbers.
        max_num_filters (int): Maximum number of filters in the encoder.
        num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
        num_downsamples_content (int): Number of times we reduce
            resolution by 2x2 for the content image.
        num_image_channels (int): Number of input image channels.
        content_norm_type (str): Type of activation normalization in the
            content encoder.
        decoder_norm_type (str): Type of activation normalization in the
            decoder.
        weight_norm_type (str): Type of weight normalization.
        output_nonlinearity (str): Type of nonlinearity before final output,
            ``'tanh'`` or ``'none'``.
        pre_act (bool): If ``True``, uses pre-activation residual blocks.
        apply_noise (bool): If ``True``, injects Gaussian noise in the decoder.
    """

    def __init__(self,
                 num_filters=64,
                 max_num_filters=256,
                 num_res_blocks=4,
                 num_downsamples_content=2,
                 num_image_channels=3,
                 content_norm_type='instance',
                 decoder_norm_type='instance',
                 weight_norm_type='',
                 output_nonlinearity='',
                 pre_act=False,
                 apply_noise=False,
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type':
                warnings.warn(
                    "Generator argument '{}' is not used.".format(key))
        self.content_encoder = ContentEncoder(num_downsamples_content,
                                              num_res_blocks,
                                              num_image_channels,
                                              num_filters,
                                              max_num_filters,
                                              'reflect',
                                              content_norm_type,
                                              weight_norm_type,
                                              'relu',
                                              pre_act)
        self.decoder = Decoder(num_downsamples_content,
                               num_res_blocks,
                               self.content_encoder.output_dim,
                               num_image_channels,
                               'reflect',
                               decoder_norm_type,
                               weight_norm_type,
                               'relu',
                               output_nonlinearity,
                               pre_act,
                               apply_noise)

    def forward(self, images):
        r"""Reconstruct an image.

        Args:
            images (Tensor): Input images.
        Returns:
            images_recon (Tensor): Reconstructed images.
        """
        content = self.content_encoder(images)
        images_recon = self.decoder(content)
        return images_recon


class ContentEncoder(nn.Module):
    r"""Improved UNIT encoder. The network consists of:

    - input layers
    - $(num_downsamples) convolutional blocks
    - $(num_res_blocks) residual blocks.
    - output layer.

    Args:
        num_downsamples (int): Number of times we reduce
            resolution by 2x2.
        num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
        num_image_channels (int): Number of input image channels.
        num_filters (int): Base filter numbers.
        max_num_filters (int): Maximum number of filters in the encoder.
        padding_mode (string): Type of padding.
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
        nonlinearity (str): Type of nonlinear activation function.
        pre_act (bool): If ``True``, uses pre-activation residual blocks.
    """

    def __init__(self,
                 num_downsamples,
                 num_res_blocks,
                 num_image_channels,
                 num_filters,
                 max_num_filters,
                 padding_mode,
                 activation_norm_type,
                 weight_norm_type,
                 nonlinearity,
                 pre_act=False):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity)
        # Whether or not it is safe to use inplace nonlinear activation.
        if not pre_act or (activation_norm_type != '' and
                           activation_norm_type != 'none'):
            conv_params['inplace_nonlinearity'] = True

        # The order of operations in residual blocks.
        order = 'pre_act' if pre_act else 'CNACNA'

        model = []
        model += [Conv2dBlock(num_image_channels, num_filters, 7, 1, 3,
                              **conv_params)]

        # Downsampling blocks.
        for i in range(num_downsamples):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            model += [Conv2dBlock(num_filters_prev, num_filters, 4, 2, 1,
                                  **conv_params)]

        # Residual blocks.
        for _ in range(num_res_blocks):
            model += [Res2dBlock(num_filters, num_filters,
                                 **conv_params,
                                 order=order)]
        self.model = nn.Sequential(*model)
        self.output_dim = num_filters

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input image.
        """
        return self.model(x)


class Decoder(nn.Module):
    r"""Improved UNIT decoder. The network consists of:

    - $(num_res_blocks) residual blocks.
    - $(num_upsamples) residual blocks or convolutional blocks
    - output layer.

    Args:
        num_upsamples (int): Number of times we increase resolution by 2x2.
        num_res_blocks (int): Number of residual blocks.
        num_filters (int): Base filter numbers.
        num_image_channels (int): Number of input image channels.
        padding_mode (string): Type of padding.
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
        nonlinearity (str): Type of nonlinear activation function.
        output_nonlinearity (str): Type of nonlinearity before final output,
            ``'tanh'`` or ``'none'``.
        pre_act (bool): If ``True``, uses pre-activation residual blocks.
        apply_noise (bool): If ``True``, injects Gaussian noise.
    """

    def __init__(self,
                 num_upsamples,
                 num_res_blocks,
                 num_filters,
                 num_image_channels,
                 padding_mode,
                 activation_norm_type,
                 weight_norm_type,
                 nonlinearity,
                 output_nonlinearity,
                 pre_act=False,
                 apply_noise=False):
        super().__init__()

        conv_params = dict(padding_mode=padding_mode,
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True,
                           apply_noise=apply_noise,
                           weight_norm_type=weight_norm_type,
                           activation_norm_type=activation_norm_type)

        # The order of operations in residual blocks.
        order = 'pre_act' if pre_act else 'CNACNA'

        # Residual blocks.
        self.decoder = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.decoder += [Res2dBlock(num_filters, num_filters,
                                        **conv_params,
                                        order=order)]

        # Convolutional blocks with upsampling.
        for i in range(num_upsamples):
            self.decoder += [NearestUpsample(scale_factor=2)]
            self.decoder += [Conv2dBlock(num_filters, num_filters // 2,
                                         5, 1, 2, **conv_params)]
            num_filters //= 2
        self.decoder += [Conv2dBlock(num_filters, num_image_channels, 7, 1, 3,
                                     nonlinearity=output_nonlinearity,
                                     padding_mode=padding_mode)]

    def forward(self, x):
        r"""

        Args:
            x (tensor): Content embedding of the content image.
        """
        for block in self.decoder:
            x = block(x)
        return x
