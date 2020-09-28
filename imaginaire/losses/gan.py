# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
import torch.nn as nn
import torch.nn.functional as F

from imaginaire.utils.distributed import master_only_print as print


@torch.jit.script
def fuse_math_min_mean_pos(x):
    r"""Fuse operation min mean for hinge loss computation of positive
    samples"""
    minval = torch.min(x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


@torch.jit.script
def fuse_math_min_mean_neg(x):
    r"""Fuse operation min mean for hinge loss computation of negative
    samples"""
    minval = torch.min(-x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


class GANLoss(nn.Module):
    r"""GAN loss constructor.

    Args:
        gan_mode (str): Type of GAN loss. ``'hinge'``, ``'least_square'``,
            ``'non_saturated'``, ``'wasserstein'``.
        target_real_label (float): The desired output label for real images.
        target_fake_label (float): The desired output label for fake images.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.gan_mode = gan_mode
        print('GAN mode: %s' % gan_mode)

    def forward(self, dis_output, t_real, dis_update=True):
        r"""GAN loss computation.

        Args:
            dis_output (tensor or list of tensors): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): If ``True``, the loss will be used to update the
                discriminator, otherwise the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if isinstance(dis_output, list):
            # For multi-scale discriminators.
            # In this implementation, the loss is first averaged for each scale
            # (batch size and number of locations) then averaged across scales,
            # so that the gradient is not dominated by the discriminator that
            # has the most output values (highest resolution).
            loss = 0
            for dis_output_i in dis_output:
                assert isinstance(dis_output_i, torch.Tensor)
                loss += self.loss(dis_output_i, t_real, dis_update)
            return loss / len(dis_output)
        else:
            return self.loss(dis_output, t_real, dis_update)

    def loss(self, dis_output, t_real, dis_update=True):
        r"""GAN loss computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if not dis_update:
            assert t_real, \
                "The target should be real when updating the generator."

        if self.gan_mode == 'non_saturated':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = F.binary_cross_entropy_with_logits(dis_output,
                                                      target_tensor)
        elif self.gan_mode == 'least_square':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = 0.5 * F.mse_loss(dis_output, target_tensor)
        elif self.gan_mode == 'hinge':
            if dis_update:
                if t_real:
                    loss = fuse_math_min_mean_pos(dis_output)
                else:
                    loss = fuse_math_min_mean_neg(dis_output)
            else:
                loss = -torch.mean(dis_output)
        elif self.gan_mode == 'wasserstein':
            if t_real:
                loss = -torch.mean(dis_output)
            else:
                loss = torch.mean(dis_output)
        else:
            raise ValueError('Unexpected gan_mode {}'.format(self.gan_mode))
        return loss

    def get_target_tensor(self, dis_output, t_real):
        r"""Return the target vector for the binary cross entropy loss
        computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
        Returns:
            target (tensor): Target tensor vector.
        """
        if t_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = dis_output.new_tensor(self.real_label)
            return self.real_label_tensor.expand_as(dis_output)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = dis_output.new_tensor(self.fake_label)
            return self.fake_label_tensor.expand_as(dis_output)
