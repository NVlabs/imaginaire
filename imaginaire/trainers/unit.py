# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch import nn

from imaginaire.evaluation import compute_fid
from imaginaire.losses import GANLoss, PerceptualLoss  # GaussianKLLoss
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.meters import Meter


class Trainer(BaseTrainer):
    r"""Reimplementation of the UNIT (https://arxiv.org/abs/1703.00848)
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
        self.best_fid_a = None
        self.best_fid_b = None

    def _init_tensorboard(self):
        r"""Initialize the tensorboard."""
        # Logging frequency: self.cfg.logging_iter
        self.meters = {}
        names = ['optim/gen_lr', 'optim/dis_lr', 'time/iteration', 'time/epoch']
        for name in names:
            self.meters[name] = Meter(name)

        # Logging frequency: self.cfg.snapshot_save_iter
        names = ['FID_a', 'best_FID_a', 'FID_b', 'best_FID_b']
        self.metric_meters = {}
        for name in names:
            self.metric_meters[name] = Meter(name)

        # Logging frequency: self.cfg.image_display_iter
        self.image_meter = Meter('images')

    def _init_loss(self, cfg):
        r"""Initialize loss terms. In UNIT, we have several loss terms
        including the GAN loss, the image reconstruction loss, the cycle
        reconstruction loss, and the gaussian kl loss. We also have an
        optional perceptual loss. A user can choose to have the gradient
        penalty loss too.

        Args:
            cfg (obj): Global configuration.
        """
        self.criteria['gan'] = GANLoss(cfg.trainer.gan_mode)
        # self.criteria['gaussian_kl'] = GaussianKLLoss()
        self.criteria['image_recon'] = nn.L1Loss()
        self.criteria['cycle_recon'] = nn.L1Loss()
        if getattr(cfg.trainer.loss_weight, 'perceptual', 0) > 0:
            self.criteria['perceptual'] = \
                PerceptualLoss(cfg=cfg,
                               network=cfg.trainer.perceptual_mode,
                               layers=cfg.trainer.perceptual_layers)

        for loss_name, loss_weight in cfg.trainer.loss_weight.__dict__.items():
            if loss_weight > 0:
                self.weights[loss_name] = loss_weight

    def gen_forward(self, data):
        r"""Compute the loss for UNIT generator.

        Args:
            data (dict): Training data at the current iteration.
        """
        cycle_recon = 'cycle_recon' in self.weights
        perceptual = 'perceptual' in self.weights
        net_G_output = self.net_G(data, cycle_recon=cycle_recon)
        net_D_output = self.net_D(data, net_G_output, real=False)

        self._time_before_loss()

        # GAN loss
        self.gen_losses['gan_a'] = self.criteria['gan'](
            net_D_output['out_ba'], True, dis_update=False)
        self.gen_losses['gan_b'] = self.criteria['gan'](
            net_D_output['out_ab'], True, dis_update=False)
        self.gen_losses['gan'] = \
            self.gen_losses['gan_a'] + self.gen_losses['gan_b']

        # Perceptual loss
        if perceptual:
            self.gen_losses['perceptual_a'] = \
                self.criteria['perceptual'](net_G_output['images_ab'],
                                            data['images_a'])
            self.gen_losses['perceptual_b'] = \
                self.criteria['perceptual'](net_G_output['images_ba'],
                                            data['images_b'])
            self.gen_losses['perceptual'] = \
                self.gen_losses['perceptual_a'] + \
                self.gen_losses['perceptual_b']

        # Image reconstruction loss
        self.gen_losses['image_recon'] = \
            self.criteria['image_recon'](net_G_output['images_aa'],
                                         data['images_a']) + \
            self.criteria['image_recon'](net_G_output['images_bb'],
                                         data['images_b'])

        """
        # KL loss
        self.gen_losses['gaussian_kl'] = \
            self.criteria['gaussian_kl'](net_G_output['content_mu_a']) + \
            self.criteria['gaussian_kl'](net_G_output['content_mu_b']) + \
            self.criteria['gaussian_kl'](net_G_output['content_mu_a_recon']) + \
            self.criteria['gaussian_kl'](net_G_output['content_mu_b_recon'])
        """

        # Cycle reconstruction loss
        if cycle_recon:
            self.gen_losses['cycle_recon_aba'] = \
                self.criteria['cycle_recon'](net_G_output['images_aba'],
                                             data['images_a'])
            self.gen_losses['cycle_recon_bab'] = \
                self.criteria['cycle_recon'](net_G_output['images_bab'],
                                             data['images_b'])
            self.gen_losses['cycle_recon'] = \
                self.gen_losses['cycle_recon_aba'] + \
                self.gen_losses['cycle_recon_bab']

        # Compute total loss
        total_loss = self._get_total_loss(gen_forward=True)
        return total_loss

    def dis_forward(self, data):
        r"""Compute the loss for UNIT discriminator.

        Args:
            data (dict): Training data at the current iteration.
        """
        with torch.no_grad():
            net_G_output = self.net_G(data, image_recon=False,
                                      cycle_recon=False)
        net_G_output['images_ba'].requires_grad = True
        net_G_output['images_ab'].requires_grad = True
        net_D_output = self.net_D(data, net_G_output)

        self._time_before_loss()

        # GAN loss.
        self.dis_losses['gan_a'] = \
            self.criteria['gan'](net_D_output['out_a'], True) + \
            self.criteria['gan'](net_D_output['out_ba'], False)
        self.dis_losses['gan_b'] = \
            self.criteria['gan'](net_D_output['out_b'], True) + \
            self.criteria['gan'](net_D_output['out_ab'], False)
        self.dis_losses['gan'] = \
            self.dis_losses['gan_a'] + self.dis_losses['gan_b']

        # Compute total loss
        total_loss = self._get_total_loss(gen_forward=False)
        return total_loss

    def _get_visualizations(self, data):
        r"""Compute visualization image.

        Args:
            data (dict): The current batch.
        """
        if self.cfg.trainer.model_average:
            net_G_for_evaluation = self.net_G.module.averaged_model
        else:
            net_G_for_evaluation = self.net_G
        with torch.no_grad():
            net_G_output = net_G_for_evaluation(data)
            vis_images = [data['images_a'],
                          data['images_b'],
                          net_G_output['images_aa'],
                          net_G_output['images_bb'],
                          net_G_output['images_ab'],
                          net_G_output['images_ba'],
                          net_G_output['images_aba'],
                          net_G_output['images_bab']]
            return vis_images

    def write_metrics(self):
        r"""Compute metrics and save them to tensorboard"""
        cur_fid_a, cur_fid_b = self._compute_fid()
        if self.best_fid_a is not None:
            self.best_fid_a = min(self.best_fid_a, cur_fid_a)
        else:
            self.best_fid_a = cur_fid_a
        if self.best_fid_b is not None:
            self.best_fid_b = min(self.best_fid_b, cur_fid_b)
        else:
            self.best_fid_b = cur_fid_b
        self._write_to_meters({'FID_a': cur_fid_a,
                               'best_FID_a': self.best_fid_a,
                               'FID_b': cur_fid_b,
                               'best_FID_b': self.best_fid_b},
                              self.metric_meters)
        self._flush_meters(self.metric_meters)

    def _compute_fid(self):
        r"""Compute FID for both domains.
        """
        self.net_G.eval()
        if self.cfg.trainer.model_average:
            net_G_for_evaluation = self.net_G.module.averaged_model
        else:
            net_G_for_evaluation = self.net_G
        fid_a_path = self._get_save_path('fid_a', 'npy')
        fid_b_path = self._get_save_path('fid_b', 'npy')
        fid_value_a = compute_fid(fid_a_path, self.val_data_loader,
                                  net_G_for_evaluation, 'images_a', 'images_ba')
        fid_value_b = compute_fid(fid_b_path, self.val_data_loader,
                                  net_G_for_evaluation, 'images_b', 'images_ab')
        print('Epoch {:05}, Iteration {:09}, FID a {}, FID b {}'.format(
            self.current_epoch, self.current_iteration,
            fid_value_a, fid_value_b))
        return fid_value_a, fid_value_b
