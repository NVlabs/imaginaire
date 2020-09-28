# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools

from torch import nn
from torch.nn import Upsample as NearestUpsample
from torch.utils.checkpoint import checkpoint

from .conv import (Conv1dBlock, Conv2dBlock, Conv3dBlock, HyperConv2dBlock,
                   LinearBlock, MultiOutConv2dBlock, PartialConv2dBlock,
                   PartialConv3dBlock)


class _BaseResBlock(nn.Module):
    r"""An abstract class for residual blocks.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity, apply_noise,
                 hidden_channels_equal_out_channels,
                 order, block, learn_shortcut):
        super().__init__()
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
        self.learn_shortcut = (in_channels != out_channels) or learn_shortcut
        if len(order) > 6 or len(order) < 5:
            raise ValueError('order must be either 5 or 6 characters')
        if hidden_channels_equal_out_channels:
            hidden_channels = out_channels
        else:
            hidden_channels = min(in_channels, out_channels)

        # Parameters that are specific for convolutions.
        conv_main_params = {}
        conv_skip_params = {}
        if block != LinearBlock:
            conv_base_params = dict(stride=1, dilation=dilation,
                                    groups=groups, padding_mode=padding_mode)
            conv_main_params.update(conv_base_params)
            conv_main_params.update(
                dict(kernel_size=kernel_size,
                     activation_norm_type=activation_norm_type,
                     activation_norm_params=activation_norm_params,
                     padding=padding))
            conv_skip_params.update(conv_base_params)
            conv_skip_params.update(dict(kernel_size=1))
            if skip_activation_norm:
                conv_skip_params.update(
                    dict(activation_norm_type=activation_norm_type,
                         activation_norm_params=activation_norm_params))

        # Other parameters.
        other_params = dict(weight_norm_type=weight_norm_type,
                            weight_norm_params=weight_norm_params,
                            apply_noise=apply_noise)

        # Residual branch.
        if order.find('A') < order.find('C') and \
                (activation_norm_type == '' or activation_norm_type == 'none'):
            # Nonlinearity is the first operation in the residual path.
            # In-place nonlinearity will modify the input variable and cause
            # backward error.
            first_inplace = False
        else:
            first_inplace = inplace_nonlinearity
        self.conv_block_0 = block(in_channels, hidden_channels,
                                  bias=biases[0],
                                  nonlinearity=nonlinearity,
                                  order=order[0:3],
                                  inplace_nonlinearity=first_inplace,
                                  **conv_main_params,
                                  **other_params)
        self.conv_block_1 = block(hidden_channels, out_channels,
                                  bias=biases[1],
                                  nonlinearity=nonlinearity,
                                  order=order[3:],
                                  inplace_nonlinearity=inplace_nonlinearity,
                                  **conv_main_params,
                                  **other_params)

        # Shortcut branch.
        if self.learn_shortcut:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = block(in_channels, out_channels,
                                      bias=biases[2],
                                      nonlinearity=skip_nonlinearity_type,
                                      order=order[0:3],
                                      **conv_skip_params,
                                      **other_params)

        # Whether this block expects conditional inputs.
        self.conditional = \
            getattr(self.conv_block_0, 'conditional', False) or \
            getattr(self.conv_block_1, 'conditional', False)

    def conv_blocks(self, x, *cond_inputs, **kw_cond_inputs):
        r"""Returns the output of the residual branch.

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            dx (tensor): Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs, **kw_cond_inputs)
        dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
        return dx

    def forward(self, x, *cond_inputs, do_checkpoint=False, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            do_checkpoint (bool, optional, default=``False``) If ``True``,
                trade compute for memory by checkpointing the model.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            output (tensor): Output tensor.
        """
        if do_checkpoint:
            dx = checkpoint(self.conv_blocks, x, *cond_inputs, **kw_cond_inputs)
        else:
            dx = self.conv_blocks(x, *cond_inputs, **kw_cond_inputs)

        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs, **kw_cond_inputs)
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return output


class ResLinearBlock(_BaseResBlock):
    r"""Residual block with full-connected layers.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
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
        apply_noise (bool, optional, default=False): If ``True``, add
            Gaussian noise with learnable magnitude after the
            fully-connected layer.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: fully-connected,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, None, None,
                         None, None, bias, None,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, LinearBlock, learn_shortcut)


class Res1dBlock(_BaseResBlock):
    r"""Residual block for 1D input.

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
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv1dBlock, learn_shortcut)


class Res2dBlock(_BaseResBlock):
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
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv2dBlock, learn_shortcut)


class Res3dBlock(_BaseResBlock):
    r"""Residual block for 3D input.

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
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv3dBlock, learn_shortcut)


class _BaseHyperResBlock(_BaseResBlock):
    r"""An abstract class for hyper residual blocks.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity, apply_noise,
                 hidden_channels_equal_out_channels,
                 order,
                 is_hyper_conv, is_hyper_norm, block, learn_shortcut):
        block = functools.partial(block,
                                  is_hyper_conv=is_hyper_conv,
                                  is_hyper_norm=is_hyper_norm)
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)

    def forward(self, x, *cond_inputs, conv_weights=(None,) * 3,
                norm_weights=(None,) * 3, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            conv_weights (list of tensors): Convolution weights for
                three convolutional layers respectively.
            norm_weights (list of tensors): Normalization weights for
                three convolutional layers respectively.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            output (tensor): Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs, conv_weights=conv_weights[0],
                               norm_weights=norm_weights[0])
        dx = self.conv_block_1(dx, *cond_inputs, conv_weights=conv_weights[1],
                               norm_weights=norm_weights[1])
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs,
                                           conv_weights=conv_weights[2],
                                           norm_weights=norm_weights[2])
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return output


class HyperRes2dBlock(_BaseHyperResBlock):
    r"""Hyper residual block for 2D input.

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
        is_hyper_conv (bool, optional, default=False): If ``True``, use
            ``HyperConv2d``, otherwise use ``torch.nn.Conv2d``.
        is_hyper_norm (bool, optional, default=False): If ``True``, use
            hyper normalizations.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='', weight_norm_params=None,
                 activation_norm_type='', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', is_hyper_conv=False, is_hyper_norm=False,
                 learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, is_hyper_conv, is_hyper_norm,
                         HyperConv2dBlock, learn_shortcut)


class _BaseDownResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks with downsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, pooling, down_factor, learn_shortcut):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)
        self.pooling = pooling(down_factor)

    def forward(self, x, *cond_inputs):
        r"""

        Args:
            x (tensor) : Input tensor.
            cond_inputs (list of tensors) : conditional input.
        Returns:
            output (tensor) : Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs)
        dx = self.conv_block_1(dx, *cond_inputs)
        dx = self.pooling(dx)
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs)
        else:
            x_shortcut = x
        x_shortcut = self.pooling(x_shortcut)
        output = x_shortcut + dx
        return output


class DownRes2dBlock(_BaseDownResBlock):
    r"""Residual block for 2D input with downsampling.

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
        pooling (class, optional, default=nn.AvgPool2d): Pytorch pooling
            layer to be used.
        down_factor (int, optional, default=2): Downsampling factor.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', pooling=nn.AvgPool2d, down_factor=2,
                 learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv2dBlock, pooling,
                         down_factor, learn_shortcut)


class _BaseUpResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks with upsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, upsample, up_factor, learn_shortcut):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)
        self.order = order
        self.upsample = upsample(scale_factor=up_factor)

    def forward(self, x, *cond_inputs):
        r"""Implementation of the up residual block forward function.
        If the order is 'NAC' for the first residual block, we will first
        do the activation norm and nonlinearity, in the original resolution.
        We will then upsample the activation map to a higher resolution. We
        then do the convolution.
        It is is other orders, then we first do the whole processing and
        then upsample.

        Args:
            x (tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional input.
        Returns:
            output (tensor) : Output tensor.
        """
        # In this particular upsample residual block operation, we first
        # upsample the skip connection.
        if self.learn_shortcut:
            x_shortcut = self.upsample(x)
            x_shortcut = self.conv_block_s(x_shortcut, *cond_inputs)
        else:
            x_shortcut = self.upsample(x)

        if self.order[0:3] == 'NAC':
            for ix, layer in enumerate(self.conv_block_0.layers.values()):
                if getattr(layer, 'conditional', False):
                    x = layer(x, *cond_inputs)
                else:
                    x = layer(x)
                if ix == 1:
                    x = self.upsample(x)
        else:
            x = self.conv_block_0(x, *cond_inputs)
            x = self.upsample(x)
        x = self.conv_block_1(x, *cond_inputs)

        output = x_shortcut + x
        return output


class UpRes2dBlock(_BaseUpResBlock):
    r"""Residual block for 2D input with downsampling.

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
        upsample (class, optional, default=NearestUpsample): PPytorch
            upsampling layer to be used.
        up_factor (int, optional, default=2): Upsampling factor.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', upsample=NearestUpsample, up_factor=2,
                 learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, Conv2dBlock,
                         upsample, up_factor, learn_shortcut)


class _BasePartialResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks with partial convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 multi_channel, return_mask,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, learn_shortcut):
        block = functools.partial(block,
                                  multi_channel=multi_channel,
                                  return_mask=return_mask)
        self.partial_conv = True
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)

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
              - output (tensor): Output tensor.
              - mask_out (tensor, optional): Masks the valid output region.
        """
        if self.conv_block_0.layers.conv.return_mask:
            dx, mask_out = self.conv_block_0(x, *cond_inputs,
                                             mask_in=mask_in, **kw_cond_inputs)
            dx, mask_out = self.conv_block_1(dx, *cond_inputs,
                                             mask_in=mask_out, **kw_cond_inputs)
        else:
            dx = self.conv_block_0(x, *cond_inputs,
                                   mask_in=mask_in, **kw_cond_inputs)
            dx = self.conv_block_1(dx, *cond_inputs,
                                   mask_in=mask_in, **kw_cond_inputs)
            mask_out = None

        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, mask_in=mask_in, *cond_inputs,
                                           **kw_cond_inputs)
            if type(x_shortcut) == tuple:
                x_shortcut, _ = x_shortcut
        else:
            x_shortcut = x
        output = x_shortcut + dx

        if mask_out is not None:
            return output, mask_out
        return output


class PartialRes2dBlock(_BasePartialResBlock):
    r"""Residual block for 2D input with partial convolution.

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
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False,
                 hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, PartialConv2dBlock, learn_shortcut)


class PartialRes3dBlock(_BasePartialResBlock):
    r"""Residual block for 3D input with partial convolution.

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
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, PartialConv3dBlock, learn_shortcut)


class _BaseMultiOutResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks that can returns multiple outputs.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, learn_shortcut):
        self.multiple_outputs = True
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)

    def forward(self, x, *cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
        Returns:
            (tuple):
              - output (tensor): Output tensor.
              - aux_outputs_0 (tensor): Auxiliary output of the first block.
              - aux_outputs_1 (tensor): Auxiliary output of the second block.
        """
        dx, aux_outputs_0 = self.conv_block_0(x, *cond_inputs)
        dx, aux_outputs_1 = self.conv_block_1(dx, *cond_inputs)
        if self.learn_shortcut:
            # We are not using the auxiliary outputs of self.conv_block_s.
            x_shortcut, _ = self.conv_block_s(x, *cond_inputs)
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return output, aux_outputs_0, aux_outputs_1


class MultiOutRes2dBlock(_BaseMultiOutResBlock):
    r"""Residual block for 2D input. It can return multiple outputs, if some
    layers in the block return more than one output.

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
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, MultiOutConv2dBlock, learn_shortcut)
