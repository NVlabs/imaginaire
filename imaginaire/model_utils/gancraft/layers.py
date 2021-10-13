# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import numpy as np
import torch
import torch.nn as nn


class AffineMod(nn.Module):
    r"""Learning affine modulation of activation.

    Args:
        in_features (int): Number of input features.
        style_features (int): Number of style features.
        mod_bias (bool): Whether to modulate bias.
    """

    def __init__(self,
                 in_features,
                 style_features,
                 mod_bias=True
                 ):
        super().__init__()
        self.weight_alpha = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
        self.bias_alpha = nn.Parameter(torch.full([in_features], 1, dtype=torch.float))  # init to 1
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        if mod_bias:
            self.weight_beta = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
            self.bias_beta = nn.Parameter(torch.full([in_features], 0, dtype=torch.float))

    @staticmethod
    def _linear_f(x, w, b):
        w = w.to(x.dtype)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b.to(x.dtype)
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    # x: B, ...   , Cin
    # z: B, 1, 1, , Cz
    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], 1, z_shape[-1])

        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)  # [B, ..., I]
        x = x * alpha

        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)  # [B, ..., I]
            x = x + beta

        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x


class ModLinear(nn.Module):
    r"""Linear layer with affine modulation (Based on StyleGAN2 mod demod).
    Equivalent to affine modulation following linear, but faster when the same modulation parameters are shared across
    multiple inputs.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        style_features (int): Number of style features.
        bias (bool): Apply additive bias before the activation function?
        mod_bias (bool): Whether to modulate bias.
        output_mode (bool): If True, modulate output instead of input.
        weight_gain (float): Initialization gain
    """

    def __init__(self,
                 in_features,
                 out_features,
                 style_features,
                 bias=True,
                 mod_bias=True,
                 output_mode=False,
                 weight_gain=1,
                 bias_init=0
                 ):
        super().__init__()
        weight_gain = weight_gain / np.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) * weight_gain)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_alpha = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
        self.bias_alpha = nn.Parameter(torch.full([in_features], 1, dtype=torch.float))  # init to 1
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        self.output_mode = output_mode
        if mod_bias:
            if output_mode:
                mod_bias_dims = out_features
            else:
                mod_bias_dims = in_features
            self.weight_beta = nn.Parameter(torch.randn([mod_bias_dims, style_features]) / np.sqrt(style_features))
            self.bias_beta = nn.Parameter(torch.full([mod_bias_dims], 0, dtype=torch.float))

    @staticmethod
    def _linear_f(x, w, b):
        w = w.to(x.dtype)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b.to(x.dtype)
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    # x: B, ...   , Cin
    # z: B, 1, 1, , Cz
    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], 1, z_shape[-1])

        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)  # [B, ..., I]
        w = self.weight.to(x.dtype)  # [O I]
        w = w.unsqueeze(0) * alpha  # [1 O I] * [B 1 I] = [B O I]

        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)  # [B, ..., I]
            if not self.output_mode:
                x = x + beta

        b = self.bias
        if b is not None:
            b = b.to(x.dtype)[None, None, :]
        if self.mod_bias and self.output_mode:
            if b is None:
                b = beta
            else:
                b = b + beta

        # [B ? I] @ [B I O] = [B ? O]
        if b is not None:
            x = torch.baddbmm(b, x, w.transpose(1, 2))
        else:
            x = x.bmm(w.transpose(1, 2))
        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x
