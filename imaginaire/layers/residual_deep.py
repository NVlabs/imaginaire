# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from imaginaire.third_party.upfirdn2d import BlurDownsample, BlurUpsample
from .conv import Conv2dBlock


class _BaseDeepResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity, apply_noise,
                 hidden_channels_equal_out_channels,
                 order, block, learn_shortcut, output_scale, skip_block=None,
                 blur=True, border_free=True, resample_first=True,
                 skip_weight_norm=True, hidden_channel_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale = output_scale
        self.resample_first = resample_first
        self.stride = stride
        self.blur = blur
        self.border_free = border_free
        assert not border_free
        if skip_block is None:
            skip_block = block

        if order == 'pre_act':
            order = 'NACNAC'
        if isinstance(bias, bool):
            # The bias for conv_block_0, conv_block_1, and conv_block_s.
            biases = [bias, bias, bias]
        elif isinstance(bias, list):
            if len(bias) == 3:
                biases = bias
            else:
                raise ValueError('Bias list must be 3.')
        else:
            raise ValueError('Bias must be either an integer or s list.')
        self.learn_shortcut = learn_shortcut
        if len(order) > 6 or len(order) < 5:
            raise ValueError('order must be either 5 or 6 characters')
        hidden_channels = in_channels // hidden_channel_ratio

        # Parameters.
        residual_params = {}
        shortcut_params = {}
        base_params = dict(dilation=dilation,
                           groups=groups,
                           padding_mode=padding_mode)
        residual_params.update(base_params)
        residual_params.update(
            dict(activation_norm_type=activation_norm_type,
                 activation_norm_params=activation_norm_params,
                 weight_norm_type=weight_norm_type,
                 weight_norm_params=weight_norm_params,
                 apply_noise=apply_noise)
        )
        shortcut_params.update(base_params)
        shortcut_params.update(dict(kernel_size=1))
        if skip_activation_norm:
            shortcut_params.update(
                dict(activation_norm_type=activation_norm_type,
                     activation_norm_params=activation_norm_params,
                     apply_noise=False))
        if skip_weight_norm:
            shortcut_params.update(
                dict(weight_norm_type=weight_norm_type,
                     weight_norm_params=weight_norm_params))

        # Residual branch.
        if order.find('A') < order.find('C') and \
                (activation_norm_type == '' or activation_norm_type == 'none'):
            # Nonlinearity is the first operation in the residual path.
            # In-place nonlinearity will modify the input variable and cause
            # backward error.
            first_inplace = False
        else:
            first_inplace = inplace_nonlinearity

        (first_stride, second_stride, shortcut_stride,
         first_blur, second_blur, shortcut_blur) = self._get_stride_blur()

        self.conv_block_1x1_in = block(
            in_channels, hidden_channels,
            1, 1, 0,
            bias=biases[0],
            nonlinearity=nonlinearity,
            order=order[0:3],
            inplace_nonlinearity=first_inplace,
            **residual_params
        )

        self.conv_block_0 = block(
            hidden_channels, hidden_channels,
            kernel_size=2 if self.border_free and first_stride < 1 else
            kernel_size,
            padding=padding,
            bias=biases[0],
            nonlinearity=nonlinearity,
            order=order[0:3],
            inplace_nonlinearity=inplace_nonlinearity,
            stride=first_stride,
            blur=first_blur,
            **residual_params
        )
        self.conv_block_1 = block(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=biases[1],
            nonlinearity=nonlinearity,
            order=order[3:],
            inplace_nonlinearity=inplace_nonlinearity,
            stride=second_stride,
            blur=second_blur,
            **residual_params
        )

        self.conv_block_1x1_out = block(
            hidden_channels, out_channels,
            1, 1, 0,
            bias=biases[1],
            nonlinearity=nonlinearity,
            order=order[0:3],
            inplace_nonlinearity=inplace_nonlinearity,
            **residual_params
        )

        # Shortcut branch.
        if self.learn_shortcut:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = skip_block(in_channels, out_channels,
                                           bias=biases[2],
                                           nonlinearity=skip_nonlinearity_type,
                                           order=order[0:3],
                                           stride=shortcut_stride,
                                           blur=shortcut_blur,
                                           **shortcut_params)
        elif in_channels < out_channels:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = skip_block(in_channels,
                                           out_channels - in_channels,
                                           bias=biases[2],
                                           nonlinearity=skip_nonlinearity_type,
                                           order=order[0:3],
                                           stride=shortcut_stride,
                                           blur=shortcut_blur,
                                           **shortcut_params)

        # Whether this block expects conditional inputs.
        self.conditional = \
            getattr(self.conv_block_0, 'conditional', False) or \
            getattr(self.conv_block_1, 'conditional', False) or \
            getattr(self.conv_block_1x1_in, 'conditional', False) or \
            getattr(self.conv_block_1x1_out, 'conditional', False)

    def _get_stride_blur(self):
        if self.stride > 1:
            # Downsampling.
            first_stride, second_stride = 1, self.stride
            first_blur, second_blur = False, self.blur
            shortcut_blur = False
            shortcut_stride = 1
            if self.blur:
                # The shortcut branch uses blur_downsample + stride-1 conv
                if self.border_free:
                    self.resample = nn.AvgPool2d(2)
                else:
                    self.resample = BlurDownsample()
            else:
                shortcut_stride = self.stride
                self.resample = nn.AvgPool2d(2)
        elif self.stride < 1:
            # Upsampling.
            first_stride, second_stride = self.stride, 1
            first_blur, second_blur = self.blur, False
            shortcut_blur = False
            shortcut_stride = 1
            if self.blur:
                # The shortcut branch uses blur_upsample + stride-1 conv
                if self.border_free:
                    self.resample = nn.Upsample(scale_factor=2,
                                                mode='bilinear')
                else:
                    self.resample = BlurUpsample()
            else:
                shortcut_stride = self.stride
                self.resample = nn.Upsample(scale_factor=2)
        else:
            first_stride = second_stride = 1
            first_blur = second_blur = False
            shortcut_stride = 1
            shortcut_blur = False
            self.resample = None
        return (first_stride, second_stride, shortcut_stride,
                first_blur, second_blur, shortcut_blur)

    def conv_blocks(
            self, x, *cond_inputs, separate_cond=False, **kw_cond_inputs
    ):
        if separate_cond:
            assert len(list(cond_inputs)) == 4
            dx = self.conv_block_1x1_in(x, cond_inputs[0],
                                        **kw_cond_inputs.get('kwargs_0', {}))
            dx = self.conv_block_0(dx, cond_inputs[1],
                                   **kw_cond_inputs.get('kwargs_1', {}))
            dx = self.conv_block_1(dx, cond_inputs[2],
                                   **kw_cond_inputs.get('kwargs_2', {}))
            dx = self.conv_block_1x1_out(dx, cond_inputs[3],
                                         **kw_cond_inputs.get('kwargs_3', {}))
        else:
            dx = self.conv_block_1x1_in(x, *cond_inputs, **kw_cond_inputs)
            dx = self.conv_block_0(dx, *cond_inputs, **kw_cond_inputs)
            dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
            dx = self.conv_block_1x1_out(dx, *cond_inputs, **kw_cond_inputs)
        return dx

    def forward(self, x, *cond_inputs, do_checkpoint=False, **kw_cond_inputs):
        if do_checkpoint:
            dx = checkpoint(self.conv_blocks, x, *cond_inputs, **kw_cond_inputs)
        else:
            dx = self.conv_blocks(x, *cond_inputs, **kw_cond_inputs)

        if self.resample_first and self.resample is not None:
            x = self.resample(x)
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(
                x, *cond_inputs, **kw_cond_inputs
            )
        elif self.in_channels < self.out_channels:
            x_shortcut_pad = self.conv_block_s(
                x, *cond_inputs, **kw_cond_inputs
            )
            x_shortcut = torch.cat((x, x_shortcut_pad), dim=1)
        elif self.in_channels > self.out_channels:
            x_shortcut = x[:, :self.out_channels, :, :]
        else:
            x_shortcut = x
        if not self.resample_first and self.resample is not None:
            x_shortcut = self.resample(x_shortcut)

        output = x_shortcut + dx
        return self.output_scale * output

    def extra_repr(self):
        s = 'output_scale={output_scale}'
        return s.format(**self.__dict__)


class DeepRes2dBlock(_BaseDeepResBlock):
    r"""Residual block for 2D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
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
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 skip_weight_norm=True,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False, output_scale=1,
                 blur=True, resample_first=True, border_free=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity, nonlinearity,
                         inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels, order, Conv2dBlock,
                         learn_shortcut, output_scale, blur=blur,
                         resample_first=resample_first, border_free=border_free,
                         skip_weight_norm=skip_weight_norm)
