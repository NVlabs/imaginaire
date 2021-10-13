# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch.nn as nn


class DictLoss(nn.Module):
    def __init__(self, criterion='l1'):
        super(DictLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)

    def forward(self, fake, real):
        """Return the target vector for the l1/l2 loss computation.

        Args:
           fake (dict, list or tuple): Discriminator features of fake images.
           real (dict, list or tuple): Discriminator features of real images.
        Returns:
           loss (tensor): Loss value.
        """
        loss = 0
        if type(fake) == dict:
            for key in fake.keys():
                loss += self.criterion(fake[key], real[key].detach())
        elif type(fake) == list or type(fake) == tuple:
            for f, r in zip(fake, real):
                loss += self.criterion(f, r.detach())
        else:
            loss += self.criterion(fake, real.detach())
        return loss
