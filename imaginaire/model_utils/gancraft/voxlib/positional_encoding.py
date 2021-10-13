# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch.autograd import Function
import voxlib

# Cheatsheet:
# mark_dirty() must be used to mark any input that is modified inplace by the forward function.
# mark_non_differentiable()


class PositionalEncodingFunction(Function):
    @staticmethod
    def forward(ctx, in_feature, pe_degrees, dim, incl_orig):
        out_feature = voxlib.positional_encoding(in_feature, pe_degrees, dim, incl_orig)

        ctx.save_for_backward(out_feature)
        ctx.pe_degrees = pe_degrees
        ctx.dim = dim
        ctx.incl_orig = incl_orig

        return out_feature

    @staticmethod
    def backward(ctx, out_feature_grad):
        out_feature, = ctx.saved_tensors

        # torch::Tensor positional_encoding_backward(const torch::Tensor& out_feature_grad,
        # const torch::Tensor& out_feature, int ndegrees, int dim, bool incl_orig) {
        in_feature_grad = voxlib.positional_encoding_backward(
            out_feature_grad, out_feature, ctx.pe_degrees, ctx.dim, ctx.incl_orig)

        return in_feature_grad, None, None, None


def positional_encoding(in_feature, pe_degrees, dim=-1, incl_orig=False):
    return PositionalEncodingFunction.apply(in_feature, pe_degrees, dim, incl_orig)

# input: N, C
# output: N, pe_degrees*C


def positional_encoding_pt(pts, pe_degrees, dim=-1, incl_orig=False):
    import numpy as np
    pe_stor = []
    for i in range(pe_degrees):
        pe_stor.append(torch.sin(pts * np.pi * 2 ** i))
        pe_stor.append(torch.cos(pts * np.pi * 2 ** i))
    if incl_orig:
        pe_stor.append(pts)
    pe = torch.cat(pe_stor, dim=dim)
    return pe


if __name__ == '__main__':
    x = torch.rand(384, 512, 5, 48).cuda() * 1024
    y = positional_encoding_pt(x, 4, incl_orig=True)
    y2 = positional_encoding(x, 4, incl_orig=True)

    print(torch.abs(y - y2))
    print(torch.allclose(y, y2, rtol=1e-05, atol=1e-05))
