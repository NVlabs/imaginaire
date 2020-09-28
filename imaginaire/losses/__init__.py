# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from .gan import GANLoss
from .perceptual import PerceptualLoss
from .feature_matching import FeatureMatchingLoss
from .kl import GaussianKLLoss
from .flow import MaskedL1Loss, FlowLoss

__all__ = ['GANLoss', 'PerceptualLoss', 'FeatureMatchingLoss', 'GaussianKLLoss',
           'MaskedL1Loss', 'FlowLoss']
