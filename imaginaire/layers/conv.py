# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .misc import ApplyNoise


class _BaseConvBlock(nn.Module):
    r"""An abstract wrapper class that wraps a torch convolution or linear layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, order, input_dim):
        super().__init__()
        from .nonlinearity import get_nonlinearity_layer
        from .weight_norm import get_weight_norm_layer
        from .activation_norm import get_activation_norm_layer
        self.weight_norm_type = weight_norm_type

        # Convolutional layer.
        if weight_norm_params is None:
            weight_norm_params = SimpleNamespace()
        weight_norm = get_weight_norm_layer(
            weight_norm_type, **vars(weight_norm_params))
        conv_layer = weight_norm(self._get_conv_layer(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, input_dim))

        # Noise injection layer.
        noise_layer = ApplyNoise() if apply_noise else None

        # Normalization layer.
        conv_before_norm = order.find('C') < order.find('N')
        norm_channels = out_channels if conv_before_norm else in_channels
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace()
        activation_norm_layer = get_activation_norm_layer(
            norm_channels,
            activation_norm_type,
            input_dim,
            **vars(activation_norm_params))

        # Nonlinearity layer.
        nonlinearity_layer = get_nonlinearity_layer(
            nonlinearity, inplace=inplace_nonlinearity)

        # Mapping from operation names to layers.
        mappings = {'C': {'conv': conv_layer},
                    'N': {'norm': activation_norm_layer},
                    'A': {'nonlinearity': nonlinearity_layer}}

        # All layers in order.
        self.layers = nn.ModuleDict()
        for op in order:
            if list(mappings[op].values())[0] is not None:
                self.layers.update(mappings[op])
                if op == 'C' and noise_layer is not None:
                    # Inject noise after convolution.
                    self.layers.update({'noise': noise_layer})

        # Whether this block expects conditional inputs.
        self.conditional = \
            getattr(conv_layer, 'conditional', False) or \
            getattr(activation_norm_layer, 'conditional', False)

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        """
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                # Layers that require conditional inputs.
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            else:
                x = layer(x)
        return x

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        # Returns the convolutional layer.
        if input_dim == 0:
            layer = nn.Linear(in_channels, out_channels, bias)
        else:
            layer_type = getattr(nn, 'Conv%dd' % input_dim)
            layer = layer_type(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias, padding_mode)
        return layer

    def __repr__(self):
        main_str = self._get_name() + '('
        child_lines = []
        for name, layer in self.layers.items():
            mod_str = repr(layer)
            if name == 'conv' and self.weight_norm_type != 'none' and \
                    self.weight_norm_type != '':
                mod_str = mod_str[:-1] + \
                    ', weight_norm={}'.format(self.weight_norm_type) + ')'
            mod_str = self._addindent(mod_str, 2)
            child_lines.append(mod_str)
        if len(child_lines) == 1:
            main_str += child_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'

        main_str += ')'
        return main_str

    @staticmethod
    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s


class LinearBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Linear`` with normalization and
    nonlinearity.

    Args:
        in_features (int): Number of channels in the input tensor.
        out_features (int): Number of channels in the output tensor.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, add
            Gaussian noise with learnable magnitude after the
            fully-connected layer.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: fully-connected,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_features, out_features, bias=True,
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_features, out_features, None, None,
                         None, None, None, bias,
                         None, weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, 0)


class Conv1dBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv1d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, 1)


class Conv2dBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, 2)


class Conv3dBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv3d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False,
                 order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, 3)


class _BaseHyperConvBlock(_BaseConvBlock):
    r"""An abstract wrapper class that wraps a hyper convolutional layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias,
                 padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity, apply_noise,
                 is_hyper_conv, is_hyper_norm,
                 order, input_dim):
        self.is_hyper_conv = is_hyper_conv
        if is_hyper_conv:
            weight_norm_type = 'none'
        if is_hyper_norm:
            activation_norm_type = 'hyper_' + activation_norm_type
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, input_dim)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        if input_dim == 0:
            raise ValueError('HyperLinearBlock is not supported.')
        else:
            name = 'HyperConv' if self.is_hyper_conv else 'nn.Conv'
            layer_type = eval(name + '%dd' % input_dim)
            layer = layer_type(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias, padding_mode)
        return layer


class HyperConv2dBlock(_BaseHyperConvBlock):
    r"""A Wrapper class that wraps ``HyperConv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        is_hyper_conv (bool, optional, default=False): If ``True``, use
            ``HyperConv2d``, otherwise use ``torch.nn.Conv2d``.
        is_hyper_norm (bool, optional, default=False): If ``True``, use
            hyper normalizations.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 is_hyper_conv=False, is_hyper_norm=False,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         is_hyper_conv, is_hyper_norm, order, 2)


class HyperConv2d(nn.Module):
    r"""Hyper Conv2d initialization.

    Args:
        in_channels (int): Dummy parameter.
        out_channels (int): Dummy parameter.
        kernel_size (int or tuple): Dummy parameter.
        stride (int or tuple, optional, default=1):
            Stride of the convolution. Default: 1
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        padding_mode (string, optional, default='zeros'):
            ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True): If ``True``,
            adds a learnable bias to the output.
    """

    def __init__(self, in_channels=0, out_channels=0, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.conditional = True

    def forward(self, x, *args, conv_weights=(None, None), **kwargs):
        r"""Hyper Conv2d forward. Convolve x using the provided weight and bias.

        Args:
            x (N x C x H x W tensor): Input tensor.
            conv_weights (N x C2 x C1 x k x k tensor or list of tensors):
                Convolution weights or [weight, bias].
        Returns:
            y (N x C2 x H x W tensor): Output tensor.
        """
        if conv_weights is None:
            conv_weight, conv_bias = None, None
        elif isinstance(conv_weights, torch.Tensor):
            conv_weight, conv_bias = conv_weights, None
        else:
            conv_weight, conv_bias = conv_weights

        if conv_weight is None:
            return x
        if conv_bias is None:
            if self.use_bias:
                raise ValueError('bias not provided but set to true during '
                                 'initialization')
            conv_bias = [None] * x.size(0)
        if self.padding_mode != 'zeros':
            x = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding

        y = None
        for i in range(x.size(0)):
            if self.stride >= 1:
                yi = F.conv2d(x[i: i + 1],
                              weight=conv_weight[i], bias=conv_bias[i],
                              stride=self.stride, padding=padding,
                              dilation=self.dilation, groups=self.groups)
            else:
                yi = F.conv_transpose2d(x[i: i + 1], weight=conv_weight[i],
                                        bias=conv_bias[i], padding=self.padding,
                                        stride=int(1 / self.stride),
                                        dilation=self.dilation,
                                        output_padding=self.padding,
                                        groups=self.groups)
            y = torch.cat([y, yi]) if y is not None else yi
        return y


class _BasePartialConvBlock(_BaseConvBlock):
    r"""An abstract wrapper class that wraps a partial convolutional layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 multi_channel, return_mask,
                 apply_noise, order, input_dim):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.partial_conv = True
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, input_dim)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        if input_dim == 2:
            layer_type = PartialConv2d
        elif input_dim == 3:
            layer_type = PartialConv3d
        else:
            raise ValueError('Partial conv only supports 2D and 3D conv now.')
        layer = layer_type(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
            multi_channel=self.multi_channel, return_mask=self.return_mask)
        return layer

    def forward(self, x, *cond_inputs, mask_in=None, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - x (tensor): Output tensor.
              - mask_out (tensor, optional): Masks the valid output region.
        """
        mask_out = None
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            elif getattr(layer, 'partial_conv', False):
                x = layer(x, mask_in=mask_in, **kw_cond_inputs)
                if type(x) == tuple:
                    x, mask_out = x
            else:
                x = layer(x)

        if mask_out is not None:
            return x, mask_out
        return x


class PartialConv2dBlock(_BasePartialConvBlock):
    r"""A Wrapper class that wraps ``PartialConv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
        multi_channel (bool, optional, default=False): If ``True``, use
            different masks for different channels.
        return_mask (bool, optional, default=True): If ``True``, the
            forward call also returns a new mask.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask, apply_noise, order, 2)


class PartialConv3dBlock(_BasePartialConvBlock):
    r"""A Wrapper class that wraps ``PartialConv3d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
        multi_channel (bool, optional, default=False): If ``True``, use
            different masks for different channels.
        return_mask (bool, optional, default=True): If ``True``, the
            forward call also returns a new mask.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask, apply_noise, order, 3)


class _MultiOutBaseConvBlock(_BaseConvBlock):
    r"""An abstract wrapper class that wraps a hyper convolutional layer with
    normalization and nonlinearity. It can return multiple outputs, if some
    layers in the block return more than one output.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias,
                 padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, order, input_dim):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, input_dim)
        self.multiple_outputs = True

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - x (tensor): Main output tensor.
              - other_outputs (list of tensors): Other output tensors.
        """
        other_outputs = []
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            if getattr(layer, 'multiple_outputs', False):
                x, other_output = layer(x)
                other_outputs.append(other_output)
            else:
                x = layer(x)
        return (x, *other_outputs)


class MultiOutConv2dBlock(_MultiOutBaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv2d`` with normalization and
    nonlinearity. It can return multiple outputs, if some layers in the block
    return more than one output.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, 2)


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
class PartialConv2d(nn.Conv2d):
    r"""Partial 2D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """

    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels,
                                                 self.in_channels,
                                                 self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0],
                                                 self.kernel_size[1])

        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None
        self.partial_conv = True

    def forward(self, x, mask_in=None):
        r"""

        Args:
            x (tensor): Input tensor.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
        """
        assert len(x.shape) == 4
        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)

                if mask_in is None:
                    # If mask is not provided, create a mask.
                    if self.multi_channel:
                        mask = torch.ones(x.data.shape[0],
                                          x.data.shape[1],
                                          x.data.shape[2],
                                          x.data.shape[3]).to(x)
                    else:
                        mask = torch.ones(1, 1, x.data.shape[2],
                                          x.data.shape[3]).to(x)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater,
                                            bias=None, stride=self.stride,
                                            padding=self.padding,
                                            dilation=self.dilation, groups=1)

                # For mixed precision training, eps from 1e-8 to 1e-6.
                eps = 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + eps)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(x, mask) if mask_in is not None else x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialConv3d(nn.Conv3d):
    r"""Partial 3D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """

    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = \
                torch.ones(self.out_channels, self.in_channels,
                           self.kernel_size[0], self.kernel_size[1],
                           self.kernel_size[2])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0],
                                                 self.kernel_size[1],
                                                 self.kernel_size[2])
        self.weight_maskUpdater = self.weight_maskUpdater.to('cuda')

        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3] * shape[4]
        self.partial_conv = True

    def forward(self, x, mask_in=None):
        r"""

        Args:
            x (tensor): Input tensor.
            mask_in (tensor, optional, default=``None``) If not ``None``, it
                masks the valid input region.
        """
        assert len(x.shape) == 5

        with torch.no_grad():
            mask = mask_in
            update_mask = F.conv3d(mask, self.weight_maskUpdater, bias=None,
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=1)

            mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, update_mask)

        raw_out = super(PartialConv3d, self).forward(torch.mul(x, mask_in))

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, mask_ratio) + bias_view
            if mask_in is not None:
                output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        if self.return_mask:
            return output, update_mask
        else:
            return output
