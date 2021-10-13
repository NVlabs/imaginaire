# flake8: noqa
import numpy as np
from types import SimpleNamespace

import torch
from torch import nn

import bias_act_cuda

# ----------------------------------------------------------------------------

activation_funcs = {
    'linear': SimpleNamespace(func=lambda x, **_: x, def_alpha=0, def_gain=1,
                              cuda_idx=1, ref='', has_2nd_grad=False),
    'relu': SimpleNamespace(func=lambda x, **_: torch.nn.functional.relu(x),
                            def_alpha=0, def_gain=np.sqrt(2), cuda_idx=2,
                            ref='y', has_2nd_grad=False),
    'leakyrelu': SimpleNamespace(
        func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha),
        def_alpha=0.2, def_gain=np.sqrt(2), cuda_idx=3, ref='y',
        has_2nd_grad=False),
    'tanh': SimpleNamespace(func=lambda x, **_: torch.tanh(x), def_alpha=0,
                            def_gain=1, cuda_idx=4, ref='y', has_2nd_grad=True),
    'sigmoid': SimpleNamespace(func=lambda x, **_: torch.sigmoid(x),
                               def_alpha=0, def_gain=1, cuda_idx=5, ref='y',
                               has_2nd_grad=True),
    'elu': SimpleNamespace(func=lambda x, **_: torch.nn.functional.elu(x),
                           def_alpha=0, def_gain=1, cuda_idx=6, ref='y',
                           has_2nd_grad=True),
    'selu': SimpleNamespace(func=lambda x, **_: torch.nn.functional.selu(x),
                            def_alpha=0, def_gain=1, cuda_idx=7, ref='y',
                            has_2nd_grad=True),
    'softplus': SimpleNamespace(
        func=lambda x, **_: torch.nn.functional.softplus(x), def_alpha=0,
        def_gain=1, cuda_idx=8, ref='y', has_2nd_grad=True),
    'swish': SimpleNamespace(func=lambda x, **_: torch.sigmoid(x) * x,
                             def_alpha=0, def_gain=np.sqrt(2), cuda_idx=9,
                             ref='x', has_2nd_grad=True),
}

# ----------------------------------------------------------------------------

_null_tensor = torch.empty([0])


def _bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None,
              impl='cuda'):
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda':
        return _bias_act_cuda(dim=dim, act=act, alpha=alpha, gain=gain,
                              clamp=clamp).apply(x, b)
    return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain,
                         clamp=clamp)


# ----------------------------------------------------------------------------

def _bias_act_ref(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)  # pylint: disable=invalid-unary-operand-type
    return x


# ----------------------------------------------------------------------------

_bias_act_cuda_cache = dict()


def _bias_act_cuda(dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """Fast CUDA implementation of `bias_act()` using custom ops.
    """
    # Parse arguments.
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Lookup from cache.
    key = (dim, act, alpha, gain, clamp)
    if key in _bias_act_cuda_cache:
        return _bias_act_cuda_cache[key]

    # Forward op.
    class BiasActCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, b):  # pylint: disable=arguments-differ
            if x.ndim > 2 and x.stride()[1] == 1:
                ctx.memory_format = torch.channels_last
            else:
                ctx.memory_format = torch.contiguous_format
            x = x.contiguous(memory_format=ctx.memory_format)
            b = b.contiguous() if b is not None else _null_tensor
            y = x
            if act != 'linear' or gain != 1 or clamp >= 0 or b is not \
                    _null_tensor:
                y = bias_act_cuda.bias_act_cuda(x, b, _null_tensor, _null_tensor,
                                                _null_tensor, 0, dim, spec.cuda_idx, alpha,
                                                gain, clamp)
            ctx.save_for_backward(
                x if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor,
                b if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor,
                y if 'y' in spec.ref else _null_tensor)
            return y

        @staticmethod
        def backward(ctx, dy):  # pylint: disable=arguments-differ
            dy = dy.contiguous(memory_format=ctx.memory_format)
            x, b, y = ctx.saved_tensors
            dx = None
            db = None

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                dx = dy
                if act != 'linear' or gain != 1 or clamp >= 0:
                    dx = BiasActCudaGrad.apply(dy, x, b, y)

            if ctx.needs_input_grad[1]:
                db = dx.sum([i for i in range(dx.ndim) if i != dim])

            return dx, db

    # Backward op.
    class BiasActCudaGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dy, x, b, y):  # pylint: disable=arguments-differ
            if x.ndim > 2 and x.stride()[1] == 1:
                ctx.memory_format = torch.channels_last
            else:
                ctx.memory_format = torch.contiguous_format
            dx = bias_act_cuda.bias_act_cuda(dy, b, x, y, _null_tensor, 1, dim,
                                             spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(
                dy if spec.has_2nd_grad else _null_tensor,
                x, b, y)
            return dx

        @staticmethod
        def backward(ctx, d_dx):  # pylint: disable=arguments-differ
            d_dx = d_dx.contiguous(memory_format=ctx.memory_format)
            dy, x, b, y = ctx.saved_tensors
            d_dy = None
            d_x = None
            d_b = None
            d_y = None

            if ctx.needs_input_grad[0]:
                d_dy = BiasActCudaGrad.apply(d_dx, x, b, y)

            if spec.has_2nd_grad and (
                    ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
                d_x = bias_act_cuda.bias_act_cuda(d_dx, b, x, y, dy, 2, dim, spec.cuda_idx,
                                                  alpha, gain, clamp)

            if spec.has_2nd_grad and ctx.needs_input_grad[2]:
                d_b = d_x.sum([i for i in range(d_x.ndim) if i != dim])

            return d_dy, d_x, d_b, d_y

    # Add to cache.
    _bias_act_cuda_cache[key] = BiasActCuda
    return BiasActCuda


class FusedNonlinearity(nn.Module):
    def __init__(self, nonlinearity, num_channels=None, lr_mul=1.0, alpha=None, impl='cuda', gain=None):
        super().__init__()
        if num_channels is not None:
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('bias', None)
        self.nonlinearity = nonlinearity
        self.gain = gain
        self.alpha = alpha
        self.lr_mul = lr_mul
        self.impl = impl

    def forward(self, x):
        bias = self.bias.type_as(x) * self.lr_mul if self.bias is not None else None
        return _bias_act(
            x, b=bias, dim=1, act=self.nonlinearity,
            alpha=self.alpha, gain=self.gain, clamp=None, impl=self.impl
        )

    def __repr__(self):
        mod_str = f'{self.__class__.__name__}(type={self.nonlinearity}'
        if self.gain is not None:
            mod_str += f', gain={self.gain}'
        if self.alpha is not None:
            mod_str += f', alpha={self.alpha}'
        if self.lr_mul != 1:
            mod_str += f', lr_mul={self.lr_mul}'
        mod_str += ')'
        return mod_str
