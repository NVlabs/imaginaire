# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch import nn

from imaginaire.generators.funit import (MLP, ContentEncoder, Decoder,
                                         StyleEncoder)


class Generator(nn.Module):
    r"""COCO-FUNIT Generator.
    """

    def __init__(self, gen_cfg, data_cfg):
        r"""COCO-FUNIT Generator constructor.

        Args:
            gen_cfg (obj): Generator definition part of the yaml config file.
            data_cfg (obj): Data definition part of the yaml config file.
        """
        super().__init__()
        self.generator = COCOFUNITTranslator(**vars(gen_cfg))

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


class COCOFUNITTranslator(nn.Module):
    r"""COCO-FUNIT Generator architecture.

    Args:
        num_filters (int): Base filter numbers.
        num_filters_mlp (int): Base filter number in the MLP module.
        style_dims (int): Dimension of the style code.
        usb_dims (int): Dimension of the universal style bias code.
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
                 usb_dims=1024,
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

        self.usb = torch.nn.Parameter(torch.randn(1, usb_dims))

        self.mlp = MLP(style_dims,
                       num_filters_mlp,
                       num_filters_mlp,
                       num_mlp_blocks,
                       'none',
                       'relu')

        num_content_mlp_blocks = 2
        num_style_mlp_blocks = 2
        self.mlp_content = MLP(self.content_encoder.output_dim,
                               style_dims,
                               num_filters_mlp,
                               num_content_mlp_blocks,
                               'none',
                               'relu')

        self.mlp_style = MLP(style_dims + usb_dims,
                             style_dims,
                             num_filters_mlp,
                             num_style_mlp_blocks,
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
        content_style_code = content.mean(3).mean(2)
        content_style_code = self.mlp_content(content_style_code)
        batch_size = style.size(0)
        usb = self.usb.repeat(batch_size, 1)
        style = style.view(batch_size, -1)
        style_in = self.mlp_style(torch.cat([style, usb], 1))
        coco_style = style_in * content_style_code
        coco_style = self.mlp(coco_style)
        images = self.decoder(content, coco_style)
        return images
