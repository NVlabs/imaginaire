# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import warnings
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import Upsample as NearestUpsample

from imaginaire.layers import Conv2dBlock, LinearBlock, Res2dBlock
from imaginaire.generators.unit import ContentEncoder


class Generator(nn.Module):
    r"""Improved MUNIT generator.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.autoencoder_a = AutoEncoder(**vars(gen_cfg))
        self.autoencoder_b = AutoEncoder(**vars(gen_cfg))

    def forward(self, data, random_style=True, image_recon=True,
                latent_recon=True, cycle_recon=True, within_latent_recon=False):
        r"""In MUNIT's forward pass, it generates a content code and a style
        code from images in both domain. It then performs a within-domain
        reconstruction step and a cross-domain translation step.
        In within-domain reconstruction, it reconstructs an image using the
        content and style from the same image and optionally encodes the image
        back to the latent space.
        In cross-domain translation, it generates an translated image by mixing
        the content and style from images in different domains, and optionally
        encodes the image back to the latent space.

        Args:
            data (dict): Training data at the current iteration.
              - images_a (tensor): Images from domain A.
              - images_b (tensor): Images from domain B.
            random_style (bool): If ``True``, samples the style code from the
                prior distribution, otherwise uses the style code encoded from
                the input images in the other domain.
            image_recon (bool): If ``True``, also returns reconstructed images.
            latent_recon (bool): If ``True``, also returns reconstructed latent
                code during cross-domain translation.
            cycle_recon (bool): If ``True``, also returns cycle
                reconstructed images.
            within_latent_recon (bool): If ``True``, also returns reconstructed
                latent code during within-domain reconstruction.
        """

        images_a = data['images_a']
        images_b = data['images_b']
        net_G_output = dict()

        # encode input images into content and style code
        content_a, style_a = self.autoencoder_a.encode(images_a)
        content_b, style_b = self.autoencoder_b.encode(images_b)

        # decode (within domain)
        if image_recon:
            images_aa = self.autoencoder_a.decode(content_a, style_a)
            images_bb = self.autoencoder_b.decode(content_b, style_b)
            net_G_output.update(dict(images_aa=images_aa, images_bb=images_bb))

        # decode (cross domain)
        if random_style:  # use randomly sampled style code
            style_a_rand = torch.randn_like(style_a)
            style_b_rand = torch.randn_like(style_b)
        else:  # use style code encoded from the other domain
            style_a_rand = style_a
            style_b_rand = style_b
        images_ba = self.autoencoder_a.decode(content_b, style_a_rand)
        images_ab = self.autoencoder_b.decode(content_a, style_b_rand)

        # encode translated images into content and style code
        if latent_recon or cycle_recon:
            content_ba, style_ba = self.autoencoder_a.encode(images_ba)
            content_ab, style_ab = self.autoencoder_b.encode(images_ab)
            net_G_output.update(dict(content_ba=content_ba, style_ba=style_ba,
                                     content_ab=content_ab, style_ab=style_ab))

        # encode reconstructed images into content and style code
        if image_recon and within_latent_recon:
            content_aa, style_aa = self.autoencoder_a.encode(images_aa)
            content_bb, style_bb = self.autoencoder_b.encode(images_bb)
            net_G_output.update(dict(content_aa=content_aa, style_aa=style_aa,
                                     content_bb=content_bb, style_bb=style_bb))

        # cycle reconstruction
        if cycle_recon:
            images_aba = self.autoencoder_a.decode(content_ab, style_a)
            images_bab = self.autoencoder_b.decode(content_ba, style_b)
            net_G_output.update(
                dict(images_aba=images_aba, images_bab=images_bab))

        # required outputs
        net_G_output.update(dict(content_a=content_a, content_b=content_b,
                                 style_a=style_a, style_b=style_b,
                                 style_a_rand=style_a_rand,
                                 style_b_rand=style_b_rand,
                                 images_ba=images_ba, images_ab=images_ab))

        return net_G_output

    def inference(self, data, a2b=True, random_style=True):
        r"""MUNIT inference.

        Args:
            data (dict): Training data at the current iteration.
              - images_a (tensor): Images from domain A.
              - images_b (tensor): Images from domain B.
            a2b (bool): If ``True``, translates images from domain A to B,
                otherwise from B to A.
            random_style (bool): If ``True``, samples the style code from the
                prior distribution, otherwise uses the style code encoded from
                the input images in the other domain.
        """
        if a2b:
            input_key = 'images_a'
            content_encode = self.autoencoder_a.content_encoder
            style_encode = self.autoencoder_b.style_encoder
            decode = self.autoencoder_b.decode
        else:
            input_key = 'images_b'
            content_encode = self.autoencoder_b.content_encoder
            style_encode = self.autoencoder_a.style_encoder
            decode = self.autoencoder_a.decode

        content_images = data[input_key]
        content = content_encode(content_images)
        if random_style:
            style_channels = self.autoencoder_a.style_channels
            style = torch.randn(content.size(0), style_channels, 1, 1,
                                device=torch.device('cuda'))
            file_names = data['key'][input_key]['filename']
        else:
            style_key = 'images_b' if a2b else 'images_a'
            assert style_key in data.keys(), \
                "{} must be provided when 'random_style' " \
                "is set to False".format(style_key)
            style_images = data[style_key]
            style = style_encode(style_images)
            file_names = \
                [content_name + '_style_' + style_name
                 for content_name, style_name in
                    zip(data['key'][input_key]['filename'],
                        data['key'][style_key]['filename'])]

        output_images = decode(content, style)
        return output_images, file_names


class AutoEncoder(nn.Module):
    r"""Improved MUNIT autoencoder.

    Args:
        num_filters (int): Base filter numbers.
        max_num_filters (int): Maximum number of filters in the encoder.
        num_filters_mlp (int): Base filter number in the MLP module.
        latent_dim (int): Dimension of the style code.
        num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
        num_mlp_blocks (int): Number of layers in the MLP module.
        num_downsamples_style (int): Number of times we reduce
            resolution by 2x2 for the style image.
        num_downsamples_content (int): Number of times we reduce
            resolution by 2x2 for the content image.
        num_image_channels (int): Number of input image channels.
        content_norm_type (str): Type of activation normalization in the
            content encoder.
        style_norm_type (str): Type of activation normalization in the
            style encoder.
        decoder_norm_type (str): Type of activation normalization in the
            decoder.
        weight_norm_type (str): Type of weight normalization.
        decoder_norm_params (obj): Parameters of activation normalization in the
            decoder. If not ``None``, decoder_norm_params.__dict__ will be used
            as keyword arguments when initializing activation normalization.
        output_nonlinearity (str): Type of nonlinearity before final output,
            ``'tanh'`` or ``'none'``.
        pre_act (bool): If ``True``, uses pre-activation residual blocks.
        apply_noise (bool): If ``True``, injects Gaussian noise in the decoder.
    """

    def __init__(self,
                 num_filters=64,
                 max_num_filters=256,
                 num_filters_mlp=256,
                 latent_dim=8,
                 num_res_blocks=4,
                 num_mlp_blocks=2,
                 num_downsamples_style=4,
                 num_downsamples_content=2,
                 num_image_channels=3,
                 content_norm_type='instance',
                 style_norm_type='',
                 decoder_norm_type='instance',
                 weight_norm_type='',
                 decoder_norm_params=SimpleNamespace(affine=False),
                 output_nonlinearity='',
                 pre_act=False,
                 apply_noise=False,
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type':
                warnings.warn(
                    "Generator argument '{}' is not used.".format(key))
        self.style_encoder = StyleEncoder(num_downsamples_style,
                                          num_image_channels,
                                          num_filters,
                                          latent_dim,
                                          'reflect',
                                          style_norm_type,
                                          weight_norm_type,
                                          'relu')
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
                               num_filters_mlp,
                               'reflect',
                               decoder_norm_type,
                               decoder_norm_params,
                               weight_norm_type,
                               'relu',
                               output_nonlinearity,
                               pre_act,
                               apply_noise)
        self.mlp = MLP(latent_dim,
                       num_filters_mlp,
                       num_filters_mlp,
                       num_mlp_blocks,
                       'none',
                       'relu')
        self.style_channels = latent_dim

    def forward(self, images):
        r"""Reconstruct an image.

        Args:
            images (Tensor): Input images.
        Returns:
            images_recon (Tensor): Reconstructed images.
        """
        content, style = self.encode(images)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        r"""Encode an image to content and style code.

        Args:
            images (Tensor): Input images.
        Returns:
            (tuple):
              - content (Tensor): Content code.
              - style (Tensor): Style code.
        """
        style = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style

    def decode(self, content, style):
        r"""Decode content and style code to an image.

        Args:
            content (Tensor): Content code.
            style (Tensor): Style code.
        Returns:
            images (Tensor): Output images.
        """
        style = self.mlp(style)
        images = self.decoder(content, style)
        return images


class StyleEncoder(nn.Module):
    r"""MUNIT style encoder.

    Args:
        num_downsamples (int): Number of times we reduce
            resolution by 2x2.
        num_image_channels (int): Number of input image channels.
        num_filters (int): Base filter numbers.
        style_channels (int): Dimension of the style code.
        padding_mode (string): Type of padding.
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
        nonlinearity (str): Type of nonlinear activation function.
    """

    def __init__(self, num_downsamples, num_image_channels, num_filters,
                 style_channels, padding_mode, activation_norm_type,
                 weight_norm_type, nonlinearity):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True)
        model = []
        model += [Conv2dBlock(num_image_channels, num_filters, 7, 1, 3,
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


class Decoder(nn.Module):
    r"""Improved MUNIT decoder. The network consists of

    - $(num_res_blocks) residual blocks.
    - $(num_upsamples) residual blocks or convolutional blocks
    - output layer.

    Args:
        num_upsamples (int): Number of times we increase resolution by 2x2.
        num_res_blocks (int): Number of residual blocks.
        num_filters (int): Base filter numbers.
        num_image_channels (int): Number of input image channels.
        style_channels (int): Dimension of the style code.
        padding_mode (string): Type of padding.
        activation_norm_type (str): Type of activation normalization.
        activation_norm_params (obj): Parameters of activation normalization.
            If not ``None``, decoder_norm_params.__dict__ will be used
            as keyword arguments when initializing activation normalization.
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
                 style_channels,
                 padding_mode,
                 activation_norm_type,
                 activation_norm_params,
                 weight_norm_type,
                 nonlinearity,
                 output_nonlinearity,
                 pre_act=False,
                 apply_noise=False):
        super().__init__()
        adain_params = SimpleNamespace(
            activation_norm_type=activation_norm_type,
            activation_norm_params=activation_norm_params,
            cond_dims=style_channels)
        conv_params = dict(padding_mode=padding_mode,
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True,
                           apply_noise=apply_noise,
                           weight_norm_type=weight_norm_type,
                           activation_norm_type='adaptive',
                           activation_norm_params=adain_params)

        # The order of operations in residual blocks.
        order = 'pre_act' if pre_act else 'CNACNA'

        # Residual blocks with AdaIN.
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


class MLP(nn.Module):
    r"""The multi-layer perceptron (MLP) that maps Gaussian style code to a
    feature vector that is given as the conditional input to AdaIN.

    Args:
        input_dim (int): Number of channels in the input tensor.
        output_dim (int): Number of channels in the output tensor.
        latent_dim (int): Number of channels in the latent features.
        num_layers (int): Number of layers in the MLP.
        norm (str): Type of activation normalization.
        nonlinearity (str): Type of nonlinear activation function.
    """

    def __init__(self, input_dim, output_dim, latent_dim, num_layers,
                 norm, nonlinearity):
        super().__init__()
        model = []
        model += [LinearBlock(input_dim, latent_dim,
                              activation_norm_type=norm,
                              nonlinearity=nonlinearity)]
        for i in range(num_layers - 2):
            model += [LinearBlock(latent_dim, latent_dim,
                                  activation_norm_type=norm,
                                  nonlinearity=nonlinearity)]
        model += [LinearBlock(latent_dim, output_dim,
                              activation_norm_type=norm,
                              nonlinearity=nonlinearity)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input image.
        """
        return self.model(x.view(x.size(0), -1))
