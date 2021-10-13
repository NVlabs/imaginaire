# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from types import SimpleNamespace

import torch
from torch import nn

from .misc import ApplyNoise
from imaginaire.third_party.upfirdn2d.upfirdn2d import Blur


class ViT2dBlock(nn.Module):
    r"""An abstract wrapper class that wraps a torch convolution or linear layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, blur, order, input_dim, clamp,
                 blur_kernel=(1, 3, 3, 1), output_scale=None,
                 init_gain=1.0):
        super().__init__()
        from .nonlinearity import get_nonlinearity_layer
        from .weight_norm import get_weight_norm_layer
        from .activation_norm import get_activation_norm_layer
        self.weight_norm_type = weight_norm_type
        self.stride = stride
        self.clamp = clamp
        self.init_gain = init_gain

        # Nonlinearity layer.
        if 'fused' in nonlinearity:
            # Fusing nonlinearity with bias.
            lr_mul = getattr(weight_norm_params, 'lr_mul', 1)
            conv_before_nonlinearity = order.find('C') < order.find('A')
            if conv_before_nonlinearity:
                assert bias
                bias = False
            channel = out_channels if conv_before_nonlinearity else in_channels
            nonlinearity_layer = get_nonlinearity_layer(
                nonlinearity, inplace=inplace_nonlinearity,
                num_channels=channel, lr_mul=lr_mul)
        else:
            nonlinearity_layer = get_nonlinearity_layer(
                nonlinearity, inplace=inplace_nonlinearity)

        # Noise injection layer.
        if apply_noise:
            order = order.replace('C', 'CG')
            noise_layer = ApplyNoise()
        else:
            noise_layer = None

        # Convolutional layer.
        if blur:
            if stride == 2:
                # Blur - Conv - Noise - Activate
                p = (len(blur_kernel) - 2) + (kernel_size - 1)
                pad0, pad1 = (p + 1) // 2, p // 2
                padding = 0
                blur_layer = Blur(
                    blur_kernel, pad=(pad0, pad1), padding_mode=padding_mode
                )
                order = order.replace('C', 'BC')
            elif stride == 0.5:
                # Conv - Blur - Noise - Activate
                padding = 0
                p = (len(blur_kernel) - 2) - (kernel_size - 1)
                pad0, pad1 = (p + 1) // 2 + 1, p // 2 + 1
                blur_layer = Blur(
                    blur_kernel, pad=(pad0, pad1), padding_mode=padding_mode
                )
                order = order.replace('C', 'CB')
            elif stride == 1:
                # No blur for now
                blur_layer = nn.Identity()
            else:
                raise NotImplementedError
        else:
            blur_layer = nn.Identity()

        if weight_norm_params is None:
            weight_norm_params = SimpleNamespace()
        weight_norm = get_weight_norm_layer(
            weight_norm_type, **vars(weight_norm_params))
        conv_layer = weight_norm(self._get_conv_layer(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, input_dim))

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

        # Mapping from operation names to layers.
        mappings = {'C': {'conv': conv_layer},
                    'N': {'norm': activation_norm_layer},
                    'A': {'nonlinearity': nonlinearity_layer}}
        mappings.update({'B': {'blur': blur_layer}})
        mappings.update({'G': {'noise': noise_layer}})

        # All layers in order.
        self.layers = nn.ModuleDict()
        for op in order:
            if list(mappings[op].values())[0] is not None:
                self.layers.update(mappings[op])

        # Whether this block expects conditional inputs.
        self.conditional = \
            getattr(conv_layer, 'conditional', False) or \
            getattr(activation_norm_layer, 'conditional', False)

        if output_scale is not None:
            self.output_scale = nn.Parameter(torch.tensor(output_scale))
        else:
            self.register_parameter("output_scale", None)

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        """
        for key, layer in self.layers.items():
            if getattr(layer, 'conditional', False):
                # Layers that require conditional inputs.
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            else:
                x = layer(x)
            if self.clamp is not None and isinstance(layer, nn.Conv2d):
                x.clamp_(max=self.clamp)
            if key == 'conv':
                if self.output_scale is not None:
                    x = x * self.output_scale
        return x

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        # Returns the convolutional layer.
        if input_dim == 0:
            layer = nn.Linear(in_channels, out_channels, bias)
        else:
            if stride < 1:  # Fractionally-strided convolution.
                padding_mode = 'zeros'
                assert padding == 0
                layer_type = getattr(nn, f'ConvTranspose{input_dim}d')
                stride = round(1 / stride)
            else:
                layer_type = getattr(nn, f'Conv{input_dim}d')
            layer = layer_type(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, groups=groups, bias=bias,
                padding_mode=padding_mode
            )

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
            if name == 'conv' and getattr(layer, 'base_lr_mul', 1) != 1:
                mod_str = mod_str[:-1] + \
                          ', lr_mul={}'.format(layer.base_lr_mul) + ')'
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
