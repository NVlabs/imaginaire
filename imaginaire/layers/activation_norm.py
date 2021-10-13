# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# flake8: noqa E722
from types import SimpleNamespace

import torch

try:
    from torch.nn import SyncBatchNorm
except ImportError:
    from torch.nn import BatchNorm2d as SyncBatchNorm
from torch import nn
from torch.nn import functional as F
from .conv import LinearBlock, Conv2dBlock, HyperConv2d, PartialConv2dBlock
from .misc import PartialSequential, ApplyNoise


class AdaptiveNorm(nn.Module):
    r"""Adaptive normalization layer. The layer first normalizes the input, then
    performs an affine transformation using parameters computed from the
    conditional inputs.

    Args:
        num_features (int): Number of channels in the input tensor.
        cond_dims (int): Number of channels in the conditional inputs.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``, or ``'weight_demod'``.
        projection (bool): If ``True``, project the conditional input to gamma
            and beta using a fully connected layer, otherwise directly use
            the conditional input as gamma and beta.
        projection_bias (bool) If ``True``, use bias in the fully connected
            projection layer.
        separate_projection (bool): If ``True``, we will use two different
            layers for gamma and beta. Otherwise, we will use one layer. It
            matters only if you apply any weight norms to this layer.
        input_dim (int): Number of dimensions of the input tensor.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
    """

    def __init__(self, num_features, cond_dims, weight_norm_type='',
                 projection=True,
                 projection_bias=True,
                 separate_projection=False,
                 input_dim=2,
                 activation_norm_type='instance',
                 activation_norm_params=None,
                 apply_noise=False,
                 add_bias=True,
                 input_scale=1.0,
                 init_gain=1.0):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              input_dim,
                                              **vars(activation_norm_params))
        if apply_noise:
            self.noise_layer = ApplyNoise()
        else:
            self.noise_layer = None

        if projection:
            if separate_projection:
                self.fc_gamma = \
                    LinearBlock(cond_dims, num_features,
                                weight_norm_type=weight_norm_type,
                                bias=projection_bias)
                self.fc_beta = \
                    LinearBlock(cond_dims, num_features,
                                weight_norm_type=weight_norm_type,
                                bias=projection_bias)
            else:
                self.fc = LinearBlock(cond_dims, num_features * 2,
                                      weight_norm_type=weight_norm_type,
                                      bias=projection_bias)

        self.projection = projection
        self.separate_projection = separate_projection
        self.input_scale = input_scale
        self.add_bias = add_bias
        self.conditional = True
        self.init_gain = init_gain

    def forward(self, x, y, noise=None, **_kwargs):
        r"""Adaptive Normalization forward.

        Args:
            x (N x C1 x * tensor): Input tensor.
            y (N x C2 tensor): Conditional information.
        Returns:
            out (N x C1 x * tensor): Output tensor.
        """
        y = y * self.input_scale
        if self.projection:
            if self.separate_projection:
                gamma = self.fc_gamma(y)
                beta = self.fc_beta(y)
                for _ in range(x.dim() - gamma.dim()):
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
            else:
                y = self.fc(y)
                for _ in range(x.dim() - y.dim()):
                    y = y.unsqueeze(-1)
                gamma, beta = y.chunk(2, 1)
        else:
            for _ in range(x.dim() - y.dim()):
                y = y.unsqueeze(-1)
            gamma, beta = y.chunk(2, 1)
        if self.norm is not None:
            x = self.norm(x)
        if self.noise_layer is not None:
            x = self.noise_layer(x, noise=noise)
        if self.add_bias:
            x = torch.addcmul(beta, x, 1 + gamma)
            return x
        else:
            return x * (1 + gamma), beta.squeeze(3).squeeze(2)


class SpatiallyAdaptiveNorm(nn.Module):
    r"""Spatially Adaptive Normalization (SPADE) initialization.

    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        separate_projection (bool): If ``True``, we will use two different
            layers for gamma and beta. Otherwise, we will use one layer. It
            matters only if you apply any weight norms to this layer.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
    """

    def __init__(self,
                 num_features,
                 cond_dims,
                 num_filters=128,
                 kernel_size=3,
                 weight_norm_type='',
                 separate_projection=False,
                 activation_norm_type='sync_batch',
                 activation_norm_params=None,
                 bias_only=False,
                 partial=False,
                 interpolation='nearest'):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        padding = kernel_size // 2
        self.separate_projection = separate_projection
        self.mlps = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()
        self.bias_only = bias_only
        self.interpolation = interpolation

        # Make cond_dims a list.
        if type(cond_dims) != list:
            cond_dims = [cond_dims]

        # Make num_filters a list.
        if not isinstance(num_filters, list):
            num_filters = [num_filters] * len(cond_dims)
        else:
            assert len(num_filters) >= len(cond_dims)

        # Make partial a list.
        if not isinstance(partial, list):
            partial = [partial] * len(cond_dims)
        else:
            assert len(partial) >= len(cond_dims)

        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            conv_block = PartialConv2dBlock if partial[i] else Conv2dBlock
            sequential = PartialSequential if partial[i] else nn.Sequential

            if num_filters[i] > 0:
                mlp += [conv_block(cond_dim,
                                   num_filters[i],
                                   kernel_size,
                                   padding=padding,
                                   weight_norm_type=weight_norm_type,
                                   nonlinearity='relu')]
            mlp_ch = cond_dim if num_filters[i] == 0 else num_filters[i]

            if self.separate_projection:
                if partial[i]:
                    raise NotImplementedError(
                        'Separate projection not yet implemented for ' +
                        'partial conv')
                self.mlps.append(nn.Sequential(*mlp))
                self.gammas.append(
                    conv_block(mlp_ch, num_features,
                               kernel_size,
                               padding=padding,
                               weight_norm_type=weight_norm_type))
                self.betas.append(
                    conv_block(mlp_ch, num_features,
                               kernel_size,
                               padding=padding,
                               weight_norm_type=weight_norm_type))
            else:
                mlp += [conv_block(mlp_ch, num_features * 2, kernel_size,
                                   padding=padding,
                                   weight_norm_type=weight_norm_type)]
                self.mlps.append(sequential(*mlp))

        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              2,
                                              **vars(activation_norm_params))
        self.conditional = True

    def forward(self, x, *cond_inputs, **_kwargs):
        r"""Spatially Adaptive Normalization (SPADE) forward.

        Args:
            x (N x C1 x H x W tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x) if self.norm is not None else x
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            label_map = F.interpolate(cond_inputs[i], size=x.size()[2:], mode=self.interpolation)
            if self.separate_projection:
                hidden = self.mlps[i](label_map)
                gamma = self.gammas[i](hidden)
                beta = self.betas[i](hidden)
            else:
                affine_params = self.mlps[i](label_map)
                gamma, beta = affine_params.chunk(2, dim=1)
            if self.bias_only:
                output = output + beta
            else:
                output = output * (1 + gamma) + beta
        return output


class DualAdaptiveNorm(nn.Module):
    def __init__(self,
                 num_features,
                 cond_dims,
                 projection_bias=True,
                 weight_norm_type='',
                 activation_norm_type='instance',
                 activation_norm_params=None,
                 apply_noise=False,
                 bias_only=False,
                 init_gain=1.0,
                 fc_scale=None,
                 is_spatial=None):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.mlps = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()
        self.bias_only = bias_only

        # Make cond_dims a list.
        if type(cond_dims) != list:
            cond_dims = [cond_dims]

        if is_spatial is None:
            is_spatial = [False for _ in range(len(cond_dims))]
        self.is_spatial = is_spatial

        for cond_dim, this_is_spatial in zip(cond_dims, is_spatial):
            kwargs = dict(weight_norm_type=weight_norm_type,
                          bias=projection_bias,
                          init_gain=init_gain,
                          output_scale=fc_scale)
            if this_is_spatial:
                self.gammas.append(Conv2dBlock(cond_dim, num_features, 1, 1, 0, **kwargs))
                self.betas.append(Conv2dBlock(cond_dim, num_features, 1, 1, 0, **kwargs))
            else:
                self.gammas.append(LinearBlock(cond_dim, num_features, **kwargs))
                self.betas.append(LinearBlock(cond_dim, num_features, **kwargs))

        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              2,
                                              **vars(activation_norm_params))
        self.conditional = True

    def forward(self, x, *cond_inputs, **_kwargs):
        assert len(cond_inputs) == len(self.gammas)
        output = self.norm(x) if self.norm is not None else x
        for cond, gamma_layer, beta_layer in zip(cond_inputs, self.gammas, self.betas):
            if cond is None:
                continue
            gamma = gamma_layer(cond)
            beta = beta_layer(cond)
            if cond.dim() == 4 and gamma.shape != x.shape:
                gamma = F.interpolate(gamma, size=x.size()[2:], mode='bilinear')
                beta = F.interpolate(beta, size=x.size()[2:], mode='bilinear')
            elif cond.dim() == 2:
                gamma = gamma[:, :, None, None]
                beta = beta[:, :, None, None]
            if self.bias_only:
                output = output + beta
            else:
                output = output * (1 + gamma) + beta
        return output


class HyperSpatiallyAdaptiveNorm(nn.Module):
    r"""Spatially Adaptive Normalization (SPADE) initialization.

    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the conditional input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``.
        is_hyper (bool): Whether to use hyper SPADE.
    """

    def __init__(self, num_features, cond_dims,
                 num_filters=0, kernel_size=3,
                 weight_norm_type='',
                 activation_norm_type='sync_batch', is_hyper=True):
        super().__init__()
        padding = kernel_size // 2
        self.mlps = nn.ModuleList()
        if type(cond_dims) != list:
            cond_dims = [cond_dims]

        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            if not is_hyper or (i != 0):
                if num_filters > 0:
                    mlp += [Conv2dBlock(cond_dim, num_filters, kernel_size,
                                        padding=padding,
                                        weight_norm_type=weight_norm_type,
                                        nonlinearity='relu')]
                mlp_ch = cond_dim if num_filters == 0 else num_filters
                mlp += [Conv2dBlock(mlp_ch, num_features * 2, kernel_size,
                                    padding=padding,
                                    weight_norm_type=weight_norm_type)]
                mlp = nn.Sequential(*mlp)
            else:
                if num_filters > 0:
                    raise ValueError('Multi hyper layer not supported yet.')
                mlp = HyperConv2d(padding=padding)
            self.mlps.append(mlp)

        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              2,
                                              affine=False)

        self.conditional = True

    def forward(self, x, *cond_inputs,
                norm_weights=(None, None), **_kwargs):
        r"""Spatially Adaptive Normalization (SPADE) forward.

        Args:
            x (4D tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
            norm_weights (5D tensor or list of tensors): conv weights or
            [weights, biases].
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x)
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            if type(cond_inputs[i]) == list:
                cond_input, mask = cond_inputs[i]
                mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear', align_corners=False)
            else:
                cond_input = cond_inputs[i]
                mask = None
            label_map = F.interpolate(cond_input, size=x.size()[2:])
            if norm_weights is None or norm_weights[0] is None or i != 0:
                affine_params = self.mlps[i](label_map)
            else:
                affine_params = self.mlps[i](label_map,
                                             conv_weights=norm_weights)
            gamma, beta = affine_params.chunk(2, dim=1)
            if mask is not None:
                gamma = gamma * (1 - mask)
                beta = beta * (1 - mask)
            output = output * (1 + gamma) + beta
        return output


class LayerNorm2d(nn.Module):
    r"""Layer Normalization as introduced in
    https://arxiv.org/abs/1607.06450.
    This is the usual way to apply layer normalization in CNNs.
    Note that unlike the pytorch implementation which applies per-element
    scale and bias, here it applies per-channel scale and bias, similar to
    batch/instance normalization.

    Args:
        num_features (int): Number of channels in the input tensor.
        eps (float, optional, default=1e-5): a value added to the
            denominator for numerical stability.
        affine (bool, optional, default=False): If ``True``, performs
            affine transformation after normalization.
    """

    def __init__(self, num_features, eps=1e-5, channel_only=False, affine=True):
        super(LayerNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.channel_only = channel_only

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).fill_(1.0))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensor.
        """
        shape = [-1] + [1] * (x.dim() - 1)
        if self.channel_only:
            mean = x.mean(1, keepdim=True)
            std = x.std(1, keepdim=True)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ScaleNorm(nn.Module):
    r"""Scale normalization:
    "Transformers without Tears: Improving the Normalization of Self-Attention"
    Modified from:
    https://github.com/tnq177/transformers_without_tears
    """

    def __init__(self, dim=-1, learned_scale=True, eps=1e-5):
        super().__init__()
        # scale = num_features ** 0.5
        if learned_scale:
            self.scale = nn.Parameter(torch.tensor(1.))
        else:
            self.scale = 1.
        # self.num_features = num_features
        self.dim = dim
        self.eps = eps
        self.learned_scale = learned_scale

    def forward(self, x):
        # noinspection PyArgumentList
        scale = self.scale * torch.rsqrt(torch.mean(x ** 2, dim=self.dim, keepdim=True) + self.eps)
        return x * scale

    def extra_repr(self):
        s = 'learned_scale={learned_scale}'
        return s.format(**self.__dict__)


class PixelNorm(ScaleNorm):
    def __init__(self, learned_scale=False, eps=1e-5, **_kwargs):
        super().__init__(1, learned_scale, eps)


class SplitMeanStd(nn.Module):
    def __init__(self, num_features, eps=1e-5, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.multiple_outputs = True

    def forward(self, x):
        b, c, h, w = x.size()
        mean = x.view(b, c, -1).mean(-1)[:, :, None, None]
        var = x.view(b, c, -1).var(-1)[:, :, None, None]
        std = torch.sqrt(var + self.eps)

        # x = (x - mean) / std
        return x, torch.cat((mean, std), dim=1)


class ScaleNorm(nn.Module):
    r"""Scale normalization:
    "Transformers without Tears: Improving the Normalization of Self-Attention"
    Modified from:
    https://github.com/tnq177/transformers_without_tears
    """

    def __init__(self, dim=-1, learned_scale=True, eps=1e-5):
        super().__init__()
        # scale = num_features ** 0.5
        if learned_scale:
            self.scale = nn.Parameter(torch.tensor(1.))
        else:
            self.scale = 1.
        # self.num_features = num_features
        self.dim = dim
        self.eps = eps
        self.learned_scale = learned_scale

    def forward(self, x):
        # noinspection PyArgumentList
        scale = self.scale * torch.rsqrt(
            torch.mean(x ** 2, dim=self.dim, keepdim=True) + self.eps)
        return x * scale

    def extra_repr(self):
        s = 'learned_scale={learned_scale}'
        return s.format(**self.__dict__)


class PixelLayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            return self.norm(x.permute(0, 2, 3, 1).view(-1, c)).view(b, h, w, c).permute(0, 3, 1, 2)
        else:
            return self.norm(x)


def get_activation_norm_layer(num_features, norm_type, input_dim, **norm_params):
    r"""Return an activation normalization layer.

    Args:
        num_features (int): Number of feature channels.
        norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        input_dim (int): Number of input dimensions.
        norm_params: Arbitrary keyword arguments that will be used to
            initialize the activation normalization.
    """
    input_dim = max(input_dim, 1)  # Norm1d works with both 0d and 1d inputs

    if norm_type == 'none' or norm_type == '':
        norm_layer = None
    elif norm_type == 'batch':
        norm = getattr(nn, 'BatchNorm%dd' % input_dim)
        norm_layer = norm(num_features, **norm_params)
    elif norm_type == 'instance':
        affine = norm_params.pop('affine', True)  # Use affine=True by default
        norm = getattr(nn, 'InstanceNorm%dd' % input_dim)
        norm_layer = norm(num_features, affine=affine, **norm_params)
    elif norm_type == 'sync_batch':
        norm_layer = SyncBatchNorm(num_features, **norm_params)
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm(num_features, **norm_params)
    elif norm_type == 'layer_2d':
        norm_layer = LayerNorm2d(num_features, **norm_params)
    elif norm_type == 'pixel_layer':
        elementwise_affine = norm_params.pop('affine', True)  # Use affine=True by default
        norm_layer = PixelLayerNorm(num_features, elementwise_affine=elementwise_affine, **norm_params)
    elif norm_type == 'scale':
        norm_layer = ScaleNorm(**norm_params)
    elif norm_type == 'pixel':
        norm_layer = PixelNorm(**norm_params)
        import imaginaire.config
        if imaginaire.config.USE_JIT:
            norm_layer = torch.jit.script(norm_layer)
    elif norm_type == 'group':
        num_groups = norm_params.pop('num_groups', 4)
        norm_layer = nn.GroupNorm(num_channels=num_features, num_groups=num_groups, **norm_params)
    elif norm_type == 'adaptive':
        norm_layer = AdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'dual_adaptive':
        norm_layer = DualAdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'spatially_adaptive':
        if input_dim != 2:
            raise ValueError('Spatially adaptive normalization layers '
                             'only supports 2D input')
        norm_layer = SpatiallyAdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'hyper_spatially_adaptive':
        if input_dim != 2:
            raise ValueError('Spatially adaptive normalization layers '
                             'only supports 2D input')
        norm_layer = HyperSpatiallyAdaptiveNorm(num_features, **norm_params)
    else:
        raise ValueError('Activation norm layer %s '
                         'is not recognized' % norm_type)
    return norm_layer
