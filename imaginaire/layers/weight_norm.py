# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import collections
import functools

import torch
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm
from torch.nn.utils.spectral_norm import SpectralNorm, \
    SpectralNormStateDictHook, SpectralNormLoadStateDictPreHook

from .conv import LinearBlock


class WeightDemodulation(nn.Module):
    r"""Weight demodulation in
    "Analyzing and Improving the Image Quality of StyleGAN", Karras et al.

    Args:
        conv (torch.nn.Modules): Convolutional layer.
        cond_dims (int): The number of channels in the conditional input.
        eps (float, optional, default=1e-8): a value added to the
            denominator for numerical stability.
        adaptive_bias (bool, optional, default=False): If ``True``, adaptively
            predicts bias from the conditional input.
        demod (bool, optional, default=False): If ``True``, performs
            weight demodulation.
    """

    def __init__(self, conv, cond_dims, eps=1e-8,
                 adaptive_bias=False, demod=True):
        super().__init__()
        self.conv = conv
        self.adaptive_bias = adaptive_bias
        if adaptive_bias:
            self.conv.register_parameter('bias', None)
            self.fc_beta = LinearBlock(cond_dims, self.conv.out_channels)
        self.fc_gamma = LinearBlock(cond_dims, self.conv.in_channels)
        self.eps = eps
        self.demod = demod
        self.conditional = True

    def forward(self, x, y, **_kwargs):
        r"""Weight demodulation forward"""
        b, c, h, w = x.size()
        self.conv.groups = b
        gamma = self.fc_gamma(y)
        gamma = gamma[:, None, :, None, None]
        weight = self.conv.weight[None, :, :, :, :] * gamma

        if self.demod:
            d = torch.rsqrt(
                (weight ** 2).sum(
                    dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d

        x = x.reshape(1, -1, h, w)
        _, _, *ws = weight.shape
        weight = weight.reshape(b * self.conv.out_channels, *ws)
        x = self.conv._conv_forward(x, weight)

        x = x.reshape(-1, self.conv.out_channels, h, w)
        if self.adaptive_bias:
            x += self.fc_beta(y)[:, :, None, None]
        return x


def weight_demod(
        conv, cond_dims=256, eps=1e-8, adaptive_bias=False, demod=True):
    r"""Weight demodulation."""
    return WeightDemodulation(conv, cond_dims, eps, adaptive_bias, demod)


class ScaledLR(object):
    def __init__(self, weight_name, bias_name):
        self.weight_name = weight_name
        self.bias_name = bias_name

    def compute_weight(self, module):
        weight = getattr(module, self.weight_name + '_ori')
        return weight * module.weight_scale

    def compute_bias(self, module):
        bias = getattr(module, self.bias_name + '_ori')
        if bias is not None:
            return bias * module.bias_scale
        else:
            return None

    @staticmethod
    def apply(module, weight_name, bias_name, lr_mul, equalized):
        assert weight_name == 'weight'
        assert bias_name == 'bias'
        fn = ScaledLR(weight_name, bias_name)
        module.register_forward_pre_hook(fn)

        if hasattr(module, bias_name):
            # module.bias is a parameter (can be None).
            bias = getattr(module, bias_name)
            delattr(module, bias_name)
            module.register_parameter(bias_name + '_ori', bias)
        else:
            # module.bias does not exist.
            bias = None
            setattr(module, bias_name + '_ori', bias)
        if bias is not None:
            setattr(module, bias_name, bias.data)
        else:
            setattr(module, bias_name, None)
        module.register_buffer('bias_scale', torch.tensor(lr_mul))

        if hasattr(module, weight_name + '_orig'):
            # The module has been wrapped with spectral normalization.
            # We only want to keep a single weight parameter.
            weight = getattr(module, weight_name + '_orig')
            delattr(module, weight_name + '_orig')
            module.register_parameter(weight_name + '_ori', weight)
            setattr(module, weight_name + '_orig', weight.data)
            # Put this hook before the spectral norm hook.
            module._forward_pre_hooks = collections.OrderedDict(
                reversed(list(module._forward_pre_hooks.items()))
            )
            module.use_sn = True
        else:
            weight = getattr(module, weight_name)
            delattr(module, weight_name)
            module.register_parameter(weight_name + '_ori', weight)
            setattr(module, weight_name, weight.data)
            module.use_sn = False

        # assert weight.dim() == 4 or weight.dim() == 2
        if equalized:
            fan_in = weight.data.size(1) * weight.data[0][0].numel()
            # Theoretically, the gain should be sqrt(2) instead of 1.
            # The official StyleGAN2 uses 1 for some reason.
            module.register_buffer(
                'weight_scale', torch.tensor(lr_mul * ((1 / fan_in) ** 0.5))
            )
        else:
            module.register_buffer('weight_scale', torch.tensor(lr_mul))

        module.lr_mul = module.weight_scale
        module.base_lr_mul = lr_mul

        return fn

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module)
        delattr(module, self.weight_name + '_ori')

        if module.use_sn:
            setattr(module, self.weight_name + '_orig', weight.detach())
        else:
            delattr(module, self.weight_name)
            module.register_parameter(self.weight_name,
                                      torch.nn.Parameter(weight.detach()))

        with torch.no_grad():
            bias = self.compute_bias(module)
        delattr(module, self.bias_name)
        delattr(module, self.bias_name + '_ori')
        if bias is not None:
            module.register_parameter(self.bias_name,
                                      torch.nn.Parameter(bias.detach()))
        else:
            module.register_parameter(self.bias_name, None)

        module.lr_mul = 1.0
        module.base_lr_mul = 1.0

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        if module.use_sn:
            # The following spectral norm hook will compute the SN of
            # "module.weight_orig" and store the normalized weight in
            # "module.weight".
            setattr(module, self.weight_name + '_orig', weight)
        else:
            setattr(module, self.weight_name, weight)
        bias = self.compute_bias(module)
        setattr(module, self.bias_name, bias)


def remove_weight_norms(module, weight_name='weight', bias_name='bias'):
    if hasattr(module, 'weight_ori') or hasattr(module, 'weight_orig'):
        for k in list(module._forward_pre_hooks.keys()):
            hook = module._forward_pre_hooks[k]
            if (isinstance(hook, ScaledLR) or isinstance(hook, SpectralNorm)):
                hook.remove(module)
                del module._forward_pre_hooks[k]

        for k, hook in module._state_dict_hooks.items():
            if isinstance(hook, SpectralNormStateDictHook) and \
                    hook.fn.name == weight_name:
                del module._state_dict_hooks[k]
                break

        for k, hook in module._load_state_dict_pre_hooks.items():
            if isinstance(hook, SpectralNormLoadStateDictPreHook) and \
                    hook.fn.name == weight_name:
                del module._load_state_dict_pre_hooks[k]
                break

    return module


def remove_equalized_lr(module, weight_name='weight', bias_name='bias'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, ScaledLR) and hook.weight_name == weight_name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("Equalized learning rate not found")

    return module


def scaled_lr(
        module, weight_name='weight', bias_name='bias', lr_mul=1.,
        equalized=False,
):
    ScaledLR.apply(module, weight_name, bias_name, lr_mul, equalized)
    return module


def get_weight_norm_layer(norm_type, **norm_params):
    r"""Return weight normalization.

    Args:
        norm_type (str):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        norm_params: Arbitrary keyword arguments that will be used to
            initialize the weight normalization.
    """
    if norm_type == 'none' or norm_type == '':  # no normalization
        return lambda x: x
    elif norm_type == 'spectral':  # spectral normalization
        return functools.partial(spectral_norm, **norm_params)
    elif norm_type == 'weight':  # weight normalization
        return functools.partial(weight_norm, **norm_params)
    elif norm_type == 'weight_demod':  # weight demodulation
        return functools.partial(weight_demod, **norm_params)
    elif norm_type == 'equalized_lr':  # equalized learning rate
        return functools.partial(scaled_lr, equalized=True, **norm_params)
    elif norm_type == 'scaled_lr':  # equalized learning rate
        return functools.partial(scaled_lr, **norm_params)
    elif norm_type == 'equalized_lr_spectral':
        lr_mul = norm_params.pop('lr_mul', 1.0)
        return lambda x: functools.partial(
            scaled_lr, equalized=True, lr_mul=lr_mul)(
            functools.partial(spectral_norm, **norm_params)(x)
        )
    elif norm_type == 'scaled_lr_spectral':
        lr_mul = norm_params.pop('lr_mul', 1.0)
        return lambda x: functools.partial(
            scaled_lr, lr_mul=lr_mul)(
            functools.partial(spectral_norm, **norm_params)(x)
        )
    else:
        raise ValueError(
            'Weight norm layer %s is not recognized' % norm_type)
