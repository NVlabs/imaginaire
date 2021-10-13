# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from torch.autograd import Function
import voxlib

"""
It takes world coordinate as input instead of block-local coordinate. Corner IDs are looked up on-the-fly to
save memory.
"""


class SparseTrilinearWorldCoordFunction(Function):
    @staticmethod
    def forward(ctx, in_feature, corner_lut_t, in_worldcoord, ign_zero):

        out_feature = voxlib.sp_trilinear_worldcoord(in_feature, corner_lut_t, in_worldcoord, ign_zero, -1)
        ctx.ign_zero = ign_zero
        ctx.save_for_backward(in_feature, corner_lut_t, in_worldcoord)

        return out_feature

    @staticmethod
    def backward(ctx, out_feature_grad):
        in_feature, corner_lut_t, in_worldcoord = ctx.saved_tensors

        assert ctx.needs_input_grad[2] is False
        in_feature_grad, = voxlib.sp_trilinear_worldcoord_backward(
            out_feature_grad, in_feature, corner_lut_t, in_worldcoord, ctx.ign_zero, False)
        return in_feature_grad, None, None, None, None


def sparse_trilinear_interp_worldcoord(in_feature, corner_lut_t, in_worldcoord, ign_zero=False):
    return SparseTrilinearWorldCoordFunction.apply(in_feature, corner_lut_t, in_worldcoord, ign_zero)
