# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import collections
import os

import torch
import torch.nn as nn

from imaginaire.config import Config
from imaginaire.generators.spade import Generator as SPADEGenerator
from imaginaire.losses import (FeatureMatchingLoss, GaussianKLLoss, PerceptualLoss)
from imaginaire.model_utils.gancraft.loss import GANLoss
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.io import get_checkpoint
from imaginaire.utils.misc import split_labels, to_device
from imaginaire.utils.trainer import ModelAverage, WrappedModel
from imaginaire.utils.visualization import tensor2label


class GauGANLoader(object):
    r"""Manages the SPADE/GauGAN model used to generate pseudo-GTs for training GANcraft.

    Args:
        gaugan_cfg (Config): SPADE configuration.
    """

    def __init__(self, gaugan_cfg):
        print('[GauGANLoader] Loading GauGAN model.')
        cfg = Config(gaugan_cfg.config)
        default_checkpoint_path = os.path.basename(gaugan_cfg.config).split('.yaml')[0] + '-' + \
            cfg.pretrained_weight + '.pt'
        checkpoint = get_checkpoint(default_checkpoint_path, cfg.pretrained_weight)
        ckpt = torch.load(checkpoint)

        net_G = WrappedModel(ModelAverage(SPADEGenerator(cfg.gen, cfg.data).to('cuda')))
        net_G.load_state_dict(ckpt['net_G'])
        self.net_GG = net_G.module.averaged_model
        self.net_GG.eval()
        self.net_GG.half()
        print('[GauGANLoader] GauGAN loading complete.')

    def eval(self, label, z=None, style_img=None):
        r"""Produce output given segmentation and other conditioning inputs.
        random style will be used if neither z nor style_img is provided.

        Args:
            label (N x C x H x W tensor): One-hot segmentation mask of shape.
            z: Style vector.
            style_img: Style image.
        """
        inputs = {'label': label[:, :-1].detach().half()}
        random_style = True

        if z is not None:
            random_style = False
            inputs['z'] = z.detach().half()
        elif style_img is not None:
            random_style = False
            inputs['images'] = style_img.detach().half()

        net_GG_output = self.net_GG(inputs, random_style=random_style)

        return net_GG_output['fake_images']


class Trainer(BaseTrainer):
    r"""Initialize GANcraft trainer.

    Args:
        cfg (Config): Global configuration.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self,
                 cfg,
                 net_G,
                 net_D,
                 opt_G,
                 opt_D,
                 sch_G,
                 sch_D,
                 train_data_loader,
                 val_data_loader):
        super(Trainer, self).__init__(cfg, net_G, net_D, opt_G,
                                      opt_D, sch_G, sch_D,
                                      train_data_loader, val_data_loader)

        # Load the pseudo-GT network only if in training mode, else not needed.
        if not self.is_inference:
            self.gaugan_model = GauGANLoader(cfg.trainer.gaugan_loader)

    def _init_loss(self, cfg):
        r"""Initialize loss terms.

        Args:
            cfg (obj): Global configuration.
        """
        if hasattr(cfg.trainer.loss_weight, 'gan'):
            self.criteria['GAN'] = GANLoss()
            self.weights['GAN'] = cfg.trainer.loss_weight.gan
        if hasattr(cfg.trainer.loss_weight, 'pseudo_gan'):
            self.criteria['PGAN'] = GANLoss()
            self.weights['PGAN'] = cfg.trainer.loss_weight.pseudo_gan
        if hasattr(cfg.trainer.loss_weight, 'l2'):
            self.criteria['L2'] = nn.MSELoss()
            self.weights['L2'] = cfg.trainer.loss_weight.l2
        if hasattr(cfg.trainer.loss_weight, 'l1'):
            self.criteria['L1'] = nn.L1Loss()
            self.weights['L1'] = cfg.trainer.loss_weight.l1
        if hasattr(cfg.trainer, 'perceptual_loss'):
            self.criteria['Perceptual'] = \
                PerceptualLoss(
                    network=cfg.trainer.perceptual_loss.mode,
                    layers=cfg.trainer.perceptual_loss.layers,
                    weights=cfg.trainer.perceptual_loss.weights)
            self.weights['Perceptual'] = cfg.trainer.loss_weight.perceptual
        # Setup the feature matching loss.
        if hasattr(cfg.trainer.loss_weight, 'feature_matching'):
            self.criteria['FeatureMatching'] = FeatureMatchingLoss()
            self.weights['FeatureMatching'] = \
                cfg.trainer.loss_weight.feature_matching
        # Setup the Gaussian KL divergence loss.
        if hasattr(cfg.trainer.loss_weight, 'kl'):
            self.criteria['GaussianKL'] = GaussianKLLoss()
            self.weights['GaussianKL'] = cfg.trainer.loss_weight.kl

    def _start_of_epoch(self, current_epoch):
        torch.cuda.empty_cache()  # Prevent the first iteration from running OOM.

    def _start_of_iteration(self, data, current_iteration):
        r"""Model specific custom start of iteration process. We will do two
        things. First, put all the data to GPU. Second, we will resize the
        input so that it becomes multiple of the factor for bug-free
        convolutional operations. This factor is given by the yaml file.
        E.g., base = getattr(self.net_G, 'base', 32)

        Args:
            data (dict): The current batch.
            current_iteration (int): The iteration number of the current batch.
        """
        data = to_device(data, 'cuda')

        # Sample camera poses and pseudo-GTs.
        with torch.no_grad():
            samples = self.net_G.module.sample_camera(data, self.gaugan_model.eval)

        return {**data, **samples}

    def gen_forward(self, data):
        r"""Compute the loss for SPADE generator.

        Args:
            data (dict): Training data at the current iteration.
        """
        net_G_output = self.net_G(data, random_style=False)

        self._time_before_loss()

        if 'GAN' in self.criteria or 'PGAN' in self.criteria:
            incl_pseudo_real = False
            if 'FeatureMatching' in self.criteria:
                incl_pseudo_real = True
            net_D_output = self.net_D(data, net_G_output, incl_real=False, incl_pseudo_real=incl_pseudo_real)
            output_fake = net_D_output['fake_outputs']  # Choose from real_outputs and fake_outputs.

            gan_loss = self.criteria['GAN'](output_fake, True, dis_update=False)
            if 'GAN' in self.criteria:
                self.gen_losses['GAN'] = gan_loss
            if 'PGAN' in self.criteria:
                self.gen_losses['PGAN'] = gan_loss

        if 'FeatureMatching' in self.criteria:
            self.gen_losses['FeatureMatching'] = self.criteria['FeatureMatching'](
                net_D_output['fake_features'], net_D_output['pseudo_real_features'])

        if 'GaussianKL' in self.criteria:
            self.gen_losses['GaussianKL'] = self.criteria['GaussianKL'](net_G_output['mu'], net_G_output['logvar'])

        # Perceptual loss is always between fake image and pseudo real image.
        if 'Perceptual' in self.criteria:
            self.gen_losses['Perceptual'] = self.criteria['Perceptual'](
                net_G_output['fake_images'], data['pseudo_real_img'])

        # Reconstruction loss between fake and pseudo real.
        if 'L2' in self.criteria:
            self.gen_losses['L2'] = self.criteria['L2'](net_G_output['fake_images'], data['pseudo_real_img'])
        if 'L1' in self.criteria:
            self.gen_losses['L1'] = self.criteria['L1'](net_G_output['fake_images'], data['pseudo_real_img'])

        total_loss = 0
        for key in self.criteria:
            total_loss = total_loss + self.gen_losses[key] * self.weights[key]

        self.gen_losses['total'] = total_loss
        return total_loss

    def dis_forward(self, data):
        r"""Compute the loss for GANcraft discriminator.

        Args:
            data (dict): Training data at the current iteration.
        """
        if 'GAN' not in self.criteria and 'PGAN' not in self.criteria:
            return

        with torch.no_grad():
            net_G_output = self.net_G(data, random_style=False)
            net_G_output['fake_images'] = net_G_output['fake_images'].detach()

        incl_real = False
        incl_pseudo_real = False
        if 'GAN' in self.criteria:
            incl_real = True
        if 'PGAN' in self.criteria:
            incl_pseudo_real = True
        net_D_output = self.net_D(data, net_G_output, incl_real=incl_real, incl_pseudo_real=incl_pseudo_real)

        self._time_before_loss()
        total_loss = 0
        if 'GAN' in self.criteria:
            output_fake = net_D_output['fake_outputs']
            output_real = net_D_output['real_outputs']

            fake_loss = self.criteria['GAN'](output_fake, False, dis_update=True)
            true_loss = self.criteria['GAN'](output_real, True, dis_update=True)
            self.dis_losses['GAN/fake'] = fake_loss
            self.dis_losses['GAN/true'] = true_loss
            self.dis_losses['GAN'] = fake_loss + true_loss
            total_loss = total_loss + self.dis_losses['GAN'] * self.weights['GAN']
        if 'PGAN' in self.criteria:
            output_fake = net_D_output['fake_outputs']
            output_pseudo_real = net_D_output['pseudo_real_outputs']

            fake_loss = self.criteria['PGAN'](output_fake, False, dis_update=True)
            true_loss = self.criteria['PGAN'](output_pseudo_real, True, dis_update=True)
            self.dis_losses['PGAN/fake'] = fake_loss
            self.dis_losses['PGAN/true'] = true_loss
            self.dis_losses['PGAN'] = fake_loss + true_loss
            total_loss = total_loss + self.dis_losses['PGAN'] * self.weights['PGAN']

        self.dis_losses['total'] = total_loss
        return total_loss

    def _get_visualizations(self, data):
        r"""Compute visualization image.

        Args:
            data (dict): The current batch.
        """
        with torch.no_grad():
            label_lengths = self.train_data_loader.dataset.get_label_lengths()
            labels = split_labels(data['label'], label_lengths)

            # Get visualization of the real image and segmentation mask.
            segmap = tensor2label(labels['seg_maps'], label_lengths['seg_maps'], output_normalized_tensor=True)
            segmap = torch.cat([x.unsqueeze(0) for x in segmap], 0)

            # Get output from GANcraft model
            net_G_output_randstyle = self.net_G(data, random_style=True)
            net_G_output = self.net_G(data, random_style=False)

            vis_images = [data['images'], segmap, net_G_output_randstyle['fake_images'], net_G_output['fake_images']]

            if 'fake_masks' in data:
                # Get pseudo-GT.
                labels = split_labels(data['fake_masks'], label_lengths)
                segmap = tensor2label(labels['seg_maps'], label_lengths['seg_maps'], output_normalized_tensor=True)
                segmap = torch.cat([x.unsqueeze(0) for x in segmap], 0)
                vis_images.append(segmap)

            if 'pseudo_real_img' in data:
                vis_images.append(data['pseudo_real_img'])

            if self.cfg.trainer.model_average_config.enabled:
                net_G_model_average_output = self.net_G.module.averaged_model(data, random_style=True)
                vis_images.append(net_G_model_average_output['fake_images'])
        return vis_images

    def load_checkpoint(self, cfg, checkpoint_path, resume=None, load_sch=True):
        r"""Load network weights, optimizer parameters, scheduler parameters
        from a checkpoint.

        Args:
            cfg (obj): Global configuration.
            checkpoint_path (str): Path to the checkpoint.
            resume (bool or None): If not ``None``, will determine whether or
            not to load optimizers in addition to network weights.
        """
        ret = super().load_checkpoint(cfg, checkpoint_path, resume, load_sch)

        if getattr(cfg.trainer, 'reset_opt_g_on_resume', False):
            self.opt_G.state = collections.defaultdict(dict)
            print('[GANcraft::load_checkpoint] Resetting opt_G.state')
        if getattr(cfg.trainer, 'reset_opt_d_on_resume', False):
            self.opt_D.state = collections.defaultdict(dict)
            print('[GANcraft::load_checkpoint] Resetting opt_D.state')

        return ret

    def test(self, data_loader, output_dir, inference_args):
        r"""Compute results images for a batch of input data and save the
        results in the specified folder.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch dataloader.
            output_dir (str): Target location for saving the output image.
        """
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        torch.cuda.empty_cache()
        with torch.no_grad():
            net_G.inference(output_dir, **vars(inference_args))
