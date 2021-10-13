# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from imaginaire.evaluation import compute_fid, compute_kid
from imaginaire.utils.diff_aug import apply_diff_aug
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
        self.best_kid = None
        self.use_fid = getattr(cfg.trainer, 'use_fid', False)
        self.use_kid = getattr(cfg.trainer, 'use_kid', True)
        self.kid_num_subsets = getattr(cfg.trainer, 'kid_num_subsets', 1)
        self.kid_sample_size = getattr(cfg.trainer, 'kid_sample_size', 256)
        self.kid_subset_size = getattr(cfg.trainer, 'kid_subset_size', 256)
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

        # Differentiable augmentation.
        keys = ['images_recon', 'images_trans']
        net_D_output = self.net_D(data, apply_diff_aug(
                                      net_G_output, keys, self.aug_policy))

        self._time_before_loss()

        # GAN loss
        # We use both the translation and reconstruction streams.
        if 'gan' in self.weights:
            self.gen_losses['gan'] = 0.5 * (
                    self.criteria['gan'](
                        net_D_output['fake_out_trans'],
                        True, dis_update=False) +
                    self.criteria['gan'](
                        net_D_output['fake_out_recon'],
                        True, dis_update=False))

        # Image reconstruction loss
        if 'image_recon' in self.weights:
            self.gen_losses['image_recon'] = \
                self.criteria['image_recon'](net_G_output['images_recon'],
                                             data['images_content'])

        # Feature matching loss
        if 'feature_matching' in self.weights:
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
        net_D_output = self.net_D(
            apply_diff_aug(data, ['images_style'], self.aug_policy),
            apply_diff_aug(net_G_output, ['images_trans'], self.aug_policy),
            recon=False)

        self._time_before_loss()

        self.dis_losses['gan'] = \
            self.criteria['gan'](net_D_output['real_out_style'], True) + \
            self.criteria['gan'](net_D_output['fake_out_trans'], False)

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
            _, _, h, w = net_G_output['images_recon'].size()
            if 'attn_a' in net_G_output:
                for i in range(net_G_output['attn_a'].size(1)):
                    vis_images += [
                        F.interpolate(
                            net_G_output['attn_a'][:, i:i + 1, :, :], (
                                h, w)).expand(-1, 3, -1, -1)]
                for i in range(net_G_output['attn_a'].size(1)):
                    vis_images += [
                        F.interpolate(
                            net_G_output['attn_b'][:, i:i + 1, :, :], (
                                h, w)).expand(-1, 3, -1, -1)]
            if self.cfg.trainer.model_average_config.enabled:
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
        if self.cfg.trainer.model_average_config.enabled:
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

    def _compute_kid(self):
        self.net_G.eval()
        if self.cfg.trainer.model_average_config.enabled:
            net_G_for_evaluation = self.net_G.module.averaged_model
        else:
            net_G_for_evaluation = self.net_G

        all_kid_values = []
        num_test_classes = self.val_data_loader.dataset.num_style_classes
        for class_idx in range(num_test_classes):
            kid_path = self._get_save_path(os.path.join('kid', str(class_idx)),
                                           'npy')
            self.val_data_loader.dataset.set_sample_class_idx(class_idx)

            kid_value = compute_kid(
                kid_path, self.val_data_loader, net_G_for_evaluation,
                'images_style', 'images_trans',
                num_subsets=self.kid_num_subsets,
                sample_size=self.kid_sample_size,
                subset_size=self.kid_subset_size)
            all_kid_values.append(kid_value)

        if is_master():
            mean_kid = np.mean(all_kid_values)
            print('Epoch {:05}, Iteration {:09}, Mean FID {}'.format(
                self.current_epoch, self.current_iteration, mean_kid))
            return mean_kid
        else:
            return None

    def write_metrics(self):
        r"""Write metrics to the tensorboard."""
        metric_dict = {}
        if self.use_kid:
            cur_kid = self._compute_kid()
            if cur_kid is not None:
                if self.best_kid is not None:
                    self.best_kid = min(self.best_kid, cur_kid)
                else:
                    self.best_kid = cur_kid
                metric_dict.update({'KID': cur_kid, 'best_KID': self.best_kid})
        if self.use_fid:
            cur_fid = self._compute_fid()
            if cur_fid is not None:
                if self.best_fid is not None:
                    self.best_fid = min(self.best_fid, cur_fid)
                else:
                    self.best_fid = cur_fid
                metric_dict.update({'FID': cur_fid, 'best_FID': self.best_fid})

        if is_master():
            self._write_to_meters(metric_dict, self.metric_meters)
            self._flush_meters(self.metric_meters)
