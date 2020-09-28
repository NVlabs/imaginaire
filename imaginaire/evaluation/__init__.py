# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from .fid import compute_fid
from .kid import compute_kid
from .prdc import compute_prdc

__all__ = ['compute_fid', 'compute_kid', 'compute_prdc']
