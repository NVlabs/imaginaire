# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import numpy as np
import torch
from torch import nn

from imaginaire.evaluation import compute_fid
from imaginaire.losses import GANLoss
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import is_master


class Trainer(BaseTrainer):
    r"""Reimplementation of the FUNIT (https://arxiv.org/abs/1905.01723)
    algorithm.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self, cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                 train_data_loader, val_data_loader):
        super().__init__(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                         train_data_loader, val_data_loader)

    def _init_loss(self, cfg):
        r"""Initialize loss terms. In FUNIT, we have several loss terms
        including the GAN loss, the image reconstruction loss, the feature
        matching loss, and the gradient penalty loss.

        Args:
            cfg (obj): Global configuration.
        """
        self.criteria['gan'] = GANLoss(cfg.trainer.gan_mode)
        self.criteria['image_recon'] = nn.L1Loss()
        self.criteria['feature_matching'] = nn.L1Loss()

        for loss_name, loss_weight in cfg.trainer.loss_weight.__dict__.items():
            if loss_weight > 0:
                self.weights[loss_name] = loss_weight

    def gen_forward(self, data):
        r"""Compute the loss for FUNIT generator.

        Args:
            data (dict): Training data at the current iteration.
        """

        net_G_output = self.net_G(data)
        net_D_output = self.net_D(data, net_G_output)

        self._time_before_loss()

        # GAN loss
        # We use both the translation and reconstruction streams.
        self.gen_losses['gan'] = 0.5 * (
            self.criteria['gan'](
                net_D_output['fake_out_trans'], True, dis_update=False) +
            self.criteria['gan'](
                net_D_output['fake_out_recon'], True, dis_update=False))

        # Image reconstruction loss
        self.gen_losses['image_recon'] = \
            self.criteria['image_recon'](net_G_output['images_recon'],
                                         data['images_content'])

        # Feature matching loss
        self.gen_losses['feature_matching'] = \
            self.criteria['feature_matching'](
                net_D_output['fake_features_trans'],
                net_D_output['real_features_style'])

        # Compute total loss
        total_loss = self._get_total_loss(gen_forward=True)
        return total_loss

    def dis_forward(self, data):
        r"""Compute the loss for FUNIT discriminator.

        Args:
            data (dict): Training data at the current iteration.
        """
        with torch.no_grad():
            net_G_output = self.net_G(data)
        net_G_output['images_trans'].requires_grad = True
        net_D_output = self.net_D(data, net_G_output, recon=False)

        self._time_before_loss()

        self.dis_losses['gan'] = \
            self.criteria['gan'](net_D_output['real_out_style'], True) + \
            self.criteria['gan'](net_D_output['fake_out_trans'], False)

        self.dis_losses['gp'] = torch.tensor(0., device=torch.device('cuda'))

        # Compute total loss
        total_loss = self._get_total_loss(gen_forward=False)
        return total_loss

    def _get_visualizations(self, data):
        r"""Compute visualization image.

        Args:
            data (dict): The current batch.
        """
        net_G_for_evaluation = self.net_G
        with torch.no_grad():
            net_G_output = net_G_for_evaluation(data)
            vis_images = [data['images_content'],
                          data['images_style'],
                          net_G_output['images_recon'],
                          net_G_output['images_trans']]
            if self.cfg.trainer.model_average:
                net_G_for_evaluation = self.net_G.module.averaged_model
                net_G_output = net_G_for_evaluation(data)
                vis_images += [net_G_output['images_recon'],
                               net_G_output['images_trans']]
            return vis_images

    def _compute_fid(self):
        r"""Compute FID. We will compute a FID value per test class. That is
        if you have 30 test classes, we will compute 30 different FID values.
        We will then report the mean of the FID values as the final
        performance number as described in the FUNIT paper.
        """
        self.net_G.eval()
        if self.cfg.trainer.model_average:
            net_G_for_evaluation = self.net_G.module.averaged_model
        else:
            net_G_for_evaluation = self.net_G

        all_fid_values = []
        num_test_classes = self.val_data_loader.dataset.num_style_classes
        for class_idx in range(num_test_classes):
            fid_path = self._get_save_path(os.path.join('fid', str(class_idx)),
                                           'npy')
            self.val_data_loader.dataset.set_sample_class_idx(class_idx)

            fid_value = compute_fid(fid_path, self.val_data_loader,
                                    net_G_for_evaluation, 'images_style',
                                    'images_trans')
            all_fid_values.append(fid_value)

        if is_master():
            mean_fid = np.mean(all_fid_values)
            print('Epoch {:05}, Iteration {:09}, Mean FID {}'.format(
                self.current_epoch, self.current_iteration, mean_fid))
            return mean_fid
        else:
            return None
