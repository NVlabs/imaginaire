# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch.nn as nn


class FeatureMatchingLoss(nn.Module):
    r"""Compute feature matching loss"""
    def __init__(self, criterion='l1'):
        super(FeatureMatchingLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)

    def forward(self, fake_features, real_features):
        r"""Return the target vector for the binary cross entropy loss
        computation.

        Args:
           fake_features (list of lists): Discriminator features of fake images.
           real_features (list of lists): Discriminator features of real images.

        Returns:
           (tensor): Loss value.
        """
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j],
                                          real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss
