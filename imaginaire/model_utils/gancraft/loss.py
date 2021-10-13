# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        r"""GAN loss constructor.

        Args:
            target_real_label (float): Desired output label for the real images.
            target_fake_label (float): Desired output label for the fake images.
        """
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None

    def forward(self, input_x, t_real, weight=None,
                reduce_dim=True, dis_update=True):
        r"""GAN loss computation.

        Args:
            input_x (tensor or list of tensors): Output values.
            t_real (boolean): Is this output value for real images.
            reduce_dim (boolean): Whether we reduce the dimensions first. This makes a difference when we use
            multi-resolution discriminators.
            weight (float): Weight to scale the loss value.
            dis_update (boolean): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if isinstance(input_x, list):
            loss = 0
            for pred_i in input_x:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, t_real, weight,
                                        reduce_dim, dis_update)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input_x)
        else:
            return self.loss(input_x, t_real, weight, reduce_dim, dis_update)

    def loss(self, input_x, t_real, weight=None,
             reduce_dim=True, dis_update=True):
        r"""N+1 label GAN loss computation.

        Args:
            input_x (tensor): Output values.
            t_real (boolean): Is this output value for real images.
            reduce_dim (boolean): Whether we reduce the dimensions first. This makes a difference when we use
            multi-resolution discriminators.
            weight (float): Weight to scale the loss value.
            dis_update (boolean): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        assert reduce_dim is True
        pred = input_x['pred'].clone()
        label = input_x['label'].clone()
        batch_size = pred.size(0)

        # ignore label 0
        label[:, 0, ...] = 0
        pred[:, 0, ...] = 0
        pred = F.log_softmax(pred, dim=1)
        assert pred.size(1) == (label.size(1) + 1)
        if dis_update:
            if t_real:
                pred_real = pred[:, :-1, :, :]
                loss = - label * pred_real
                loss = torch.sum(loss, dim=1, keepdim=True)
            else:
                pred_fake = pred[:, -1, None, :, :]  # N plus 1
                loss = - pred_fake
        else:
            assert t_real, "GAN loss must be aiming for real."
            pred_real = pred[:, :-1, :, :]
            loss = - label * pred_real
            loss = torch.sum(loss, dim=1, keepdim=True)

        if weight is not None:
            loss = loss * weight
        if reduce_dim:
            loss = torch.mean(loss)
        else:
            loss = loss.view(batch_size, -1).mean(dim=1)
        return loss
