# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from .conv import LinearBlock, Conv1dBlock, Conv2dBlock, Conv3dBlock, \
    HyperConv2dBlock, MultiOutConv2dBlock, \
    PartialConv2dBlock, PartialConv3dBlock
from .residual import ResLinearBlock, Res1dBlock, Res2dBlock, Res3dBlock, \
    HyperRes2dBlock, MultiOutRes2dBlock, UpRes2dBlock, DownRes2dBlock, \
    PartialRes2dBlock, PartialRes3dBlock
from .non_local import NonLocal2dBlock

__all__ = ['Conv1dBlock', 'Conv2dBlock', 'Conv3dBlock', 'LinearBlock',
           'HyperConv2dBlock', 'MultiOutConv2dBlock',
           'PartialConv2dBlock', 'PartialConv3dBlock',
           'Res1dBlock', 'Res2dBlock', 'Res3dBlock',
           'UpRes2dBlock', 'DownRes2dBlock',
           'ResLinearBlock', 'HyperRes2dBlock', 'MultiOutRes2dBlock',
           'PartialRes2dBlock', 'PartialRes3dBlock',
           'NonLocal2dBlock']
