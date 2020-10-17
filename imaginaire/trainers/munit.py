# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch

from imaginaire.evaluation import compute_fid
from imaginaire.losses import (GANLoss, GaussianKLLoss,
                               PerceptualLoss)
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.meters import Meter
from imaginaire.utils.misc import random_shift


class Trainer(BaseTrainer):
    r"""Reimplementation of the MUNIT (https://arxiv.org/abs/1804.04732)
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
        self.gan_recon = getattr(cfg.trainer, 'gan_recon', False)
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
        r"""Initialize loss terms. In MUNIT, we have several loss terms
        including the GAN loss, the image reconstruction loss, the content
        reconstruction loss, the style reconstruction loss, the cycle
        reconstruction loss. We also have an optional perceptual loss. A user
        can choose to have gradient penalty or consistency regularization too.

        Args:
            cfg (obj): Global configuration.
        """
        self.criteria['gan'] = GANLoss(cfg.trainer.gan_mode)
        self.criteria['kl'] = GaussianKLLoss()
        self.criteria['image_recon'] = torch.nn.L1Loss()
        self.criteria['content_recon'] = torch.nn.L1Loss()
        self.criteria['style_recon'] = torch.nn.L1Loss()

        if getattr(cfg.trainer.loss_weight, 'perceptual', 0) > 0:
            self.criteria['perceptual'] = \
                PerceptualLoss(cfg=cfg,
                               network=cfg.trainer.perceptual_mode,
                               layers=cfg.trainer.perceptual_layers,
                               instance_normalized=True)

        for loss_name, loss_weight in cfg.trainer.loss_weight.__dict__.items():
            if loss_weight > 0:
                self.weights[loss_name] = loss_weight

    def gen_forward(self, data):
        r"""Compute the loss for MUNIT generator.

        Args:
            data (dict): Training data at the current iteration.
        """
        cycle_recon = 'cycle_recon' in self.weights
        image_recon = 'image_recon' in self.weights
        perceptual = 'perceptual' in self.weights

        net_G_output = self.net_G(data,
                                  image_recon=image_recon,
                                  cycle_recon=cycle_recon,
                                  within_latent_recon=False)
        net_D_output = self.net_D(data, net_G_output, real=False,
                                  gan_recon=self.gan_recon)

        self._time_before_loss()

        # GAN loss
        if self.gan_recon:
            self.gen_losses['gan_a'] = \
                0.5 * (self.criteria['gan'](net_D_output['out_ba'],
                                            True, dis_update=False) +
                       self.criteria['gan'](net_D_output['out_aa'],
                                            True, dis_update=False))
            self.gen_losses['gan_b'] = \
                0.5 * (self.criteria['gan'](net_D_output['out_ab'],
                                            True, dis_update=False) +
                       self.criteria['gan'](net_D_output['out_bb'],
                                            True, dis_update=False))
        else:
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
        if image_recon:
            self.gen_losses['image_recon'] = \
                self.criteria['image_recon'](net_G_output['images_aa'],
                                             data['images_a']) + \
                self.criteria['image_recon'](net_G_output['images_bb'],
                                             data['images_b'])

        # Style reconstruction loss
        self.gen_losses['style_recon_a'] = \
            self.criteria['style_recon'](net_G_output['style_ba'],
                                         net_G_output['style_a_rand'])
        self.gen_losses['style_recon_b'] = \
            self.criteria['style_recon'](net_G_output['style_ab'],
                                         net_G_output['style_b_rand'])
        self.gen_losses['style_recon'] = \
            self.gen_losses['style_recon_a'] + self.gen_losses['style_recon_b']

        # Content reconstruction loss
        self.gen_losses['content_recon_a'] = \
            self.criteria['content_recon'](net_G_output['content_ab'],
                                           net_G_output['content_a'].detach())
        self.gen_losses['content_recon_b'] = \
            self.criteria['content_recon'](net_G_output['content_ba'],
                                           net_G_output['content_b'].detach())
        self.gen_losses['content_recon'] = \
            self.gen_losses['content_recon_a'] + \
            self.gen_losses['content_recon_b']

        # KL loss
        self.gen_losses['kl'] = \
            self.criteria['kl'](net_G_output['style_a']) + \
            self.criteria['kl'](net_G_output['style_b'])

        # Cycle reconstruction loss
        if cycle_recon:
            self.gen_losses['cycle_recon'] = \
                self.criteria['image_recon'](net_G_output['images_aba'],
                                             data['images_a']) + \
                self.criteria['image_recon'](net_G_output['images_bab'],
                                             data['images_b'])

        # Compute total loss
        total_loss = self._get_total_loss(gen_forward=True)
        return total_loss

    def dis_forward(self, data):
        r"""Compute the loss for MUNIT discriminator.

        Args:
            data (dict): Training data at the current iteration.
        """
        with torch.no_grad():
            net_G_output = self.net_G(data,
                                      image_recon=self.gan_recon,
                                      latent_recon=False,
                                      cycle_recon=False,
                                      within_latent_recon=False)
        net_G_output['images_ba'].requires_grad = True
        net_G_output['images_ab'].requires_grad = True
        net_D_output = self.net_D(data, net_G_output, gan_recon=self.gan_recon)

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

        # Gradient penalty.
        if 'gp' in self.weights:
            images_a_gp = self.criteria['gp'].get_dis_inputs(
                data['images_a'], net_G_output['images_ba'])
            images_b_gp = self.criteria['gp'].get_dis_inputs(
                data['images_b'], net_G_output['images_ab'])
            net_D_input_gp = dict(images_ab=images_b_gp, images_ba=images_a_gp)
            net_D_output_gp = self.net_D(data, net_D_input_gp, real=False)
            self.dis_losses['gp_a'] = self.criteria['gp'](
                net_D_output_gp['images_ba'], net_D_output_gp['out_ba'])
            self.dis_losses['gp_b'] = self.criteria['gp'](
                net_D_output_gp['images_ab'], net_D_output_gp['out_ab'])
            self.dis_losses['gp'] = \
                self.dis_losses['gp_a'] + self.dis_losses['gp_b']

        # Consistency regularization.
        self.dis_losses['consistency_reg'] = \
            torch.tensor(0., device=torch.device('cuda'))
        if 'consistency_reg' in self.weights:
            data_aug, net_G_output_aug = {}, {}
            data_aug['images_a'] = random_shift(data['images_a'].flip(-1))
            data_aug['images_b'] = random_shift(data['images_b'].flip(-1))
            net_G_output_aug['images_ab'] = \
                random_shift(net_G_output['images_ab'].flip(-1))
            net_G_output_aug['images_ba'] = \
                random_shift(net_G_output['images_ba'].flip(-1))
            net_D_output_aug = self.net_D(data_aug, net_G_output_aug)
            feature_names = ['fea_ba', 'fea_ab',
                             'fea_a', 'fea_b']
            for feature_name in feature_names:
                self.dis_losses['consistency_reg'] += \
                    torch.pow(net_D_output_aug[feature_name] -
                              net_D_output[feature_name], 2).mean()

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
            net_G_output = net_G_for_evaluation(data, random_style=False)
            net_G_output_random = net_G_for_evaluation(data)
            vis_images = [data['images_a'],
                          data['images_b'],
                          net_G_output['images_aa'],
                          net_G_output['images_bb'],
                          net_G_output['images_ab'],
                          net_G_output_random['images_ab'],
                          net_G_output['images_ba'],
                          net_G_output_random['images_ba'],
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
