# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from .gan import GANLoss
from .perceptual import PerceptualLoss
from .feature_matching import FeatureMatchingLoss
from .kl import GaussianKLLoss
from .flow import MaskedL1Loss, FlowLoss
from .dict import DictLoss
from .weighted_mse import WeightedMSELoss

__all__ = ['GANLoss', 'PerceptualLoss', 'FeatureMatchingLoss', 'GaussianKLLoss',
           'MaskedL1Loss', 'FlowLoss', 'DictLoss',
           'WeightedMSELoss']

try:
    from .gradient_penalty import GradientPenaltyLoss
    __all__.extend(['GradientPenaltyLoss'])
except:  # noqa
    pass
