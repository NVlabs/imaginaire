# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# flake8: noqa E722
from types import SimpleNamespace

import torch


try:
    # from torch.nn import BatchNorm2d as SyncBatchNorm
    from torch.nn import SyncBatchNorm
except ImportError:
    from torch.nn import BatchNorm2d as SyncBatchNorm
from torch import nn
from torch.nn import functional as F
from .conv import LinearBlock, Conv2dBlock, HyperConv2d, PartialConv2dBlock
from .misc import PartialSequential


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
                 separate_projection=False,
                 input_dim=2,
                 activation_norm_type='instance',
                 activation_norm_params=None):
        super().__init__()
        self.projection = projection
        self.separate_projection = separate_projection
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              input_dim,
                                              **vars(activation_norm_params))
        if self.projection:
            if self.separate_projection:
                self.fc_gamma = \
                    LinearBlock(cond_dims, num_features,
                                weight_norm_type=weight_norm_type)
                self.fc_beta = \
                    LinearBlock(cond_dims, num_features,
                                weight_norm_type=weight_norm_type)
            else:
                self.fc = LinearBlock(cond_dims, num_features * 2,
                                      weight_norm_type=weight_norm_type)

        self.conditional = True

    def forward(self, x, y, **kwargs):
        r"""Adaptive Normalization forward.

        Args:
            x (N x C1 x * tensor): Input tensor.
            y (N x C2 tensor): Conditional information.
        Returns:
            out (N x C1 x * tensor): Output tensor.
        """
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
        x = self.norm(x) if self.norm is not None else x
        out = x * (1 + gamma) + beta
        return out


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
                 partial=False):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        padding = kernel_size // 2
        self.separate_projection = separate_projection
        self.mlps = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()

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

    def forward(self, x, *cond_inputs, **kwargs):
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
            label_map = F.interpolate(cond_inputs[i], size=x.size()[2:],
                                      mode='nearest')
            if self.separate_projection:
                hidden = self.mlps[i](label_map)
                gamma = self.gammas[i](hidden)
                beta = self.betas[i](hidden)
            else:
                affine_params = self.mlps[i](label_map)
                gamma, beta = affine_params.chunk(2, dim=1)
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
                norm_weights=(None, None), **kwargs):
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
                mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear',
                                     align_corners=False)
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

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensor.
        """
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def get_activation_norm_layer(num_features, norm_type,
                              input_dim, **norm_params):
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
        # There is a bug of using amp O1 with synchronize batch norm.
        # The lines below fix it.
        affine = norm_params.pop('affine', True)
        # Always call SyncBN with affine=True
        norm_layer = SyncBatchNorm(num_features, affine=True, **norm_params)
        norm_layer.weight.requires_grad = affine
        norm_layer.bias.requires_grad = affine
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm(num_features, **norm_params)
    elif norm_type == 'layer_2d':
        norm_layer = LayerNorm2d(num_features, **norm_params)
    elif norm_type == 'group':
        norm_layer = nn.GroupNorm(num_channels=num_features, **norm_params)
    elif norm_type == 'adaptive':
        norm_layer = AdaptiveNorm(num_features, **norm_params)
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
