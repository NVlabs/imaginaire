# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial
from types import SimpleNamespace

import torch
from torch import nn

from imaginaire.layers import \
    (Conv2dBlock, LinearBlock, Res2dBlock, UpRes2dBlock)


class Generator(nn.Module):
    r"""Generator of the improved FUNIT baseline in the COCO-FUNIT paper.
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.generator = FUNITTranslator(**vars(gen_cfg))

    def forward(self, data):
        r"""In the FUNIT's forward pass, it generates a content embedding and
        a style code from the content image, and a style code from the style
        image. By mixing the content code and the style code from the content
        image, we reconstruct the input image. By mixing the content code and
        the style code from the style image, we have a translation output.

        Args:
            data (dict): Training data at the current iteration.
        """
        content_a = self.generator.content_encoder(data['images_content'])
        style_a = self.generator.style_encoder(data['images_content'])
        style_b = self.generator.style_encoder(data['images_style'])
        images_trans = self.generator.decode(content_a, style_b)
        images_recon = self.generator.decode(content_a, style_a)

        net_G_output = dict(images_trans=images_trans,
                            images_recon=images_recon)
        return net_G_output

    def inference(self, data, keep_original_size=True):
        r"""COCO-FUNIT inference.

        Args:
            data (dict): Training data at the current iteration.
              - images_content (tensor): Content images.
              - images_style (tensor): Style images.
            a2b (bool): If ``True``, translates images from domain A to B,
                otherwise from B to A.
            keep_original_size (bool): If ``True``, output image is resized
            to the input content image size.
        """
        content_a = self.generator.content_encoder(data['images_content'])
        style_b = self.generator.style_encoder(data['images_style'])
        output_images = self.generator.decode(content_a, style_b)
        if keep_original_size:
            height = data['original_h_w'][0][0]
            width = data['original_h_w'][0][1]
            # print('( H, W) = ( %d, %d)' % (height, width))
            output_images = torch.nn.functional.interpolate(
                output_images, size=[height, width])
        file_names = data['key']['images_content'][0]
        return output_images, file_names


class FUNITTranslator(nn.Module):
    r"""

    Args:
         num_filters (int): Base filter numbers.
         num_filters_mlp (int): Base filter number in the MLP module.
         style_dims (int): Dimension of the style code.
         num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
         num_mlp_blocks (int): Number of layers in the MLP module.
         num_downsamples_content (int): Number of times we reduce
            resolution by 2x2 for the content image.
         num_downsamples_style (int): Number of times we reduce
            resolution by 2x2 for the style image.
         num_image_channels (int): Number of input image channels.
         weight_norm_type (str): Type of weight normalization.
             ``'none'``, ``'spectral'``, or ``'weight'``.
    """

    def __init__(self,
                 num_filters=64,
                 num_filters_mlp=256,
                 style_dims=64,
                 num_res_blocks=2,
                 num_mlp_blocks=3,
                 num_downsamples_style=4,
                 num_downsamples_content=2,
                 num_image_channels=3,
                 weight_norm_type='',
                 **kwargs):
        super().__init__()

        self.style_encoder = StyleEncoder(num_downsamples_style,
                                          num_image_channels,
                                          num_filters,
                                          style_dims,
                                          'reflect',
                                          'none',
                                          weight_norm_type,
                                          'relu')

        self.content_encoder = ContentEncoder(num_downsamples_content,
                                              num_res_blocks,
                                              num_image_channels,
                                              num_filters,
                                              'reflect',
                                              'instance',
                                              weight_norm_type,
                                              'relu')

        self.decoder = Decoder(self.content_encoder.output_dim,
                               num_filters_mlp,
                               num_image_channels,
                               num_downsamples_content,
                               'reflect',
                               weight_norm_type,
                               'relu')

        self.mlp = MLP(style_dims,
                       num_filters_mlp,
                       num_filters_mlp,
                       num_mlp_blocks,
                       'none',
                       'relu')

    def forward(self, images):
        r"""Reconstruct the input image by combining the computer content and
        style code.

        Args:
            images (tensor): Input image tensor.
        """
        # reconstruct an image
        content, style = self.encode(images)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        r"""Encoder images to get their content and style codes.

        Args:
            images (tensor): Input image tensor.
        """
        style = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style

    def decode(self, content, style):
        r"""Generate images by combining their content and style codes.

        Args:
            content (tensor): Content code tensor.
            style (tensor): Style code tensor.
        """
        style = self.mlp(style)
        images = self.decoder(content, style)
        return images


class Decoder(nn.Module):
    r"""Improved FUNIT decoder.

    Args:
        num_enc_output_channels (int): Number of content feature channels.
        style_channels (int): Dimension of the style code.
        num_image_channels (int): Number of image channels.
        num_upsamples (int): How many times we are going to apply
            upsample residual block.
    """

    def __init__(self,
                 num_enc_output_channels,
                 style_channels,
                 num_image_channels=3,
                 num_upsamples=4,
                 padding_type='reflect',
                 weight_norm_type='none',
                 nonlinearity='relu'):
        super(Decoder, self).__init__()
        adain_params = SimpleNamespace(
            activation_norm_type='instance',
            activation_norm_params=SimpleNamespace(affine=False),
            cond_dims=style_channels)

        base_res_block = partial(Res2dBlock,
                                 kernel_size=3,
                                 padding=1,
                                 padding_mode=padding_type,
                                 nonlinearity=nonlinearity,
                                 activation_norm_type='adaptive',
                                 activation_norm_params=adain_params,
                                 weight_norm_type=weight_norm_type)

        base_up_res_block = partial(UpRes2dBlock,
                                    kernel_size=5,
                                    padding=2,
                                    padding_mode=padding_type,
                                    weight_norm_type=weight_norm_type,
                                    activation_norm_type='adaptive',
                                    activation_norm_params=adain_params,
                                    skip_activation_norm='instance',
                                    skip_nonlinearity=nonlinearity,
                                    nonlinearity=nonlinearity,
                                    hidden_channels_equal_out_channels=True)

        dims = num_enc_output_channels

        # Residual blocks with AdaIN.
        self.decoder = nn.ModuleList()
        self.decoder += [base_res_block(dims, dims)]
        self.decoder += [base_res_block(dims, dims)]
        for _ in range(num_upsamples):
            self.decoder += [base_up_res_block(dims, dims // 2)]
            dims = dims // 2
        self.decoder += [Conv2dBlock(dims,
                                     num_image_channels,
                                     kernel_size=7,
                                     stride=1,
                                     padding=3,
                                     padding_mode='reflect',
                                     nonlinearity='tanh')]

    def forward(self, x, style):
        r"""

        Args:
            x (tensor): Content embedding of the content image.
            style (tensor): Style embedding of the style image.
        """
        for block in self.decoder:
            if getattr(block, 'conditional', False):
                x = block(x, style)
            else:
                x = block(x)
        return x


class StyleEncoder(nn.Module):
    r"""Improved FUNIT Style Encoder. This is basically the same as the
    original FUNIT Style Encoder.

    Args:
        num_downsamples (int): Number of times we reduce resolution by
            2x2.
        image_channels (int): Number of input image channels.
        num_filters (int): Base filter number.
        style_channels (int): Style code dimension.
        padding_mode (str): Padding mode.
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        nonlinearity (str): Nonlinearity.
    """

    def __init__(self,
                 num_downsamples,
                 image_channels,
                 num_filters,
                 style_channels,
                 padding_mode,
                 activation_norm_type,
                 weight_norm_type,
                 nonlinearity):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True)
        model = []
        model += [Conv2dBlock(image_channels, num_filters, 7, 1, 3,
                              **conv_params)]
        for i in range(2):
            model += [Conv2dBlock(num_filters, 2 * num_filters, 4, 2, 1,
                                  **conv_params)]
            num_filters *= 2
        for i in range(num_downsamples - 2):
            model += [Conv2dBlock(num_filters, num_filters, 4, 2, 1,
                                  **conv_params)]
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(num_filters, style_channels, 1, 1, 0)]
        self.model = nn.Sequential(*model)
        self.output_dim = num_filters

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input image.
        """
        return self.model(x)


class ContentEncoder(nn.Module):
    r"""Improved FUNIT Content Encoder. This is basically the same as the
    original FUNIT content encoder.

    Args:
        num_downsamples (int): Number of times we reduce resolution by
           2x2.
        num_res_blocks (int): Number of times we append residual block
           after all the downsampling modules.
        image_channels (int): Number of input image channels.
        num_filters (int): Base filter number.
        padding_mode (str): Padding mode
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        nonlinearity (str): Nonlinearity.
    """

    def __init__(self,
                 num_downsamples,
                 num_res_blocks,
                 image_channels,
                 num_filters,
                 padding_mode,
                 activation_norm_type,
                 weight_norm_type,
                 nonlinearity):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True,
                           order='CNACNA')
        model = []
        model += [Conv2dBlock(image_channels, num_filters, 7, 1, 3,
                              **conv_params)]
        dims = num_filters
        for i in range(num_downsamples):
            model += [Conv2dBlock(dims, dims * 2, 4, 2, 1, **conv_params)]
            dims *= 2

        for _ in range(num_res_blocks):
            model += [Res2dBlock(dims, dims, **conv_params)]
        self.model = nn.Sequential(*model)
        self.output_dim = dims

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input image.
        """
        return self.model(x)


class MLP(nn.Module):
    r"""Improved FUNIT style decoder.

    Args:
        input_dim (int): Input dimension (style code dimension).
        output_dim (int): Output dimension (to be fed into the AdaIN
           layer).
        latent_dim (int): Latent dimension.
        num_layers (int): Number of layers in the MLP.
        activation_norm_type (str): Activation type.
        nonlinearity (str): Nonlinearity type.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 latent_dim,
                 num_layers,
                 activation_norm_type,
                 nonlinearity):
        super().__init__()
        model = []
        model += [LinearBlock(input_dim, latent_dim,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity)]
        # changed from num_layers - 2 to num_layers - 3.
        for i in range(num_layers - 3):
            model += [LinearBlock(latent_dim, latent_dim,
                                  activation_norm_type=activation_norm_type,
                                  nonlinearity=nonlinearity)]
        model += [LinearBlock(latent_dim, output_dim,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensor.
        """
        return self.model(x.view(x.size(0), -1))
