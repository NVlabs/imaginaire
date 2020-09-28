# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools

import torch

from imaginaire.evaluation import compute_fid
from imaginaire.losses import FeatureMatchingLoss, GANLoss, PerceptualLoss
from imaginaire.model_utils.pix2pixHD import cluster_features, get_edges
from imaginaire.trainers.spade import Trainer as SPADETrainer
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.misc import to_cuda


class Trainer(SPADETrainer):
    r"""Initialize pix2pixHD trainer.

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
        super(Trainer, self).__init__(cfg, net_G, net_D, opt_G,
                                      opt_D, sch_G, sch_D,
                                      train_data_loader, val_data_loader)

    def _assign_criteria(self, name, criterion, weight):
        r"""Assign training loss terms.

        Args:
            name (str): Loss name
            criterion (obj): Loss object.
            weight (float): Loss weight. It should be non-negative.
        """
        self.criteria[name] = criterion
        self.weights[name] = weight

    def _init_loss(self, cfg):
        r"""Initialize training loss terms. In pix2pixHD, there are three
        loss terms: GAN loss, feature matching loss, and perceptual loss.

        Args:
            cfg (obj): Global configuration.
        """
        self.criteria = dict()
        self.weights = dict()
        trainer_cfg = cfg.trainer
        loss_weight = cfg.trainer.loss_weight
        # GAN loss and feature matching loss.
        self._assign_criteria('GAN',
                              GANLoss(trainer_cfg.gan_mode),
                              loss_weight.gan)
        self._assign_criteria('FeatureMatching',
                              FeatureMatchingLoss(),
                              loss_weight.feature_matching)
        self._assign_criteria('Perceptual',
                              PerceptualLoss(
                                  cfg=cfg,
                                  network=cfg.trainer.perceptual_loss.mode,
                                  layers=cfg.trainer.perceptual_loss.layers,
                                  weights=cfg.trainer.perceptual_loss.weights),
                              loss_weight.perceptual)

    def _start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current number of iteration.
        """
        return self.pre_process(data)

    def gen_forward(self, data):
        r"""Compute the loss for pix2pixHD generator.

        Args:
            data (dict): Training data at the current iteration.
        """
        net_G_output = self.net_G(data)
        net_D_output = self.net_D(data, net_G_output)

        self._time_before_loss()

        output_fake = self._get_outputs(net_D_output, real=False)
        self.gen_losses['GAN'] = \
            self.criteria['GAN'](output_fake, True, dis_update=False)

        self.gen_losses['FeatureMatching'] = self.criteria['FeatureMatching'](
            net_D_output['fake_features'], net_D_output['real_features'])

        if hasattr(self.cfg.trainer, 'perceptual_loss'):
            self.gen_losses['Perceptual'] = self.criteria['Perceptual'](
                net_G_output['fake_images'], data['images'])

        total_loss = self.gen_losses['GAN'].new_tensor([0])
        for key in self.criteria:
            total_loss += self.gen_losses[key] * self.weights[key]

        self.gen_losses['total'] = total_loss
        return total_loss

    def dis_forward(self, data):
        r"""Compute the loss for pix2pixHD discriminator.

        Args:
            data (dict): Training data at the current iteration.
        """
        with torch.no_grad():
            net_G_output = self.net_G(data)
            net_G_output['fake_images'] = net_G_output['fake_images'].detach()
        net_D_output = self.net_D(data, net_G_output)

        self._time_before_loss()

        output_fake = self._get_outputs(net_D_output, real=False)
        output_real = self._get_outputs(net_D_output, real=True)
        fake_loss = self.criteria['GAN'](output_fake, False, dis_update=True)
        true_loss = self.criteria['GAN'](output_real, True, dis_update=True)
        self.dis_losses['GAN'] = fake_loss + true_loss
        total_loss = self.dis_losses['GAN'] * self.weights['GAN']
        self.dis_losses['total'] = total_loss
        return total_loss

    def pre_process(self, data):
        r"""Data pre-processing step for the pix2pixHD method. It takes a
        dictionary as input where the dictionary contains a label field. The
        label field is the concatenation of the segmentation mask and the
        instance map. In this function, we will replace the instance map with
        an edge map. We will also add a "instance_maps" field to the dictionary.

        Args:
            data (dict): Input dictionary.
            data['label']: Input label map where the last channel is the
                instance map.
        """
        data = to_cuda(data)
        if self.cfg.trainer.model_average:
            net_G = self.net_G.module.module
        else:
            net_G = self.net_G.module
        if net_G.contain_instance_map:
            inst_maps = data['label'][:, -1:]
            edge_maps = get_edges(inst_maps)
            data['instance_maps'] = inst_maps.clone()
            data['label'][:, -1:] = edge_maps
        return data

    def _pre_save_checkpoint(self):
        r"""Implement the things you want to do before saving the checkpoints.
        For example, you can compute the K-mean features (pix2pixHD) before
        saving the model weights to the checkponts.
        """
        if hasattr(self.cfg.gen, 'enc'):
            if self.cfg.trainer.model_average:
                net_E = self.net_G.module.averaged_model.encoder
            else:
                net_E = self.net_G.module.encoder
            is_cityscapes = getattr(self.cfg.gen, 'is_cityscapes', False)
            cluster_features(self.cfg, self.val_data_loader,
                             net_E,
                             self.pre_process,
                             is_cityscapes)

    def _compute_fid(self):
        r"""We will compute FID for the regular model using the eval mode.
        For the moving average model, we will use the eval mode.
        """
        self.net_G.eval()
        net_G_for_evaluation = \
            functools.partial(self.net_G, random_style=True)
        regular_fid_path = self._get_save_path('regular_fid', 'npy')
        regular_fid_value = compute_fid(regular_fid_path,
                                        self.val_data_loader,
                                        net_G_for_evaluation,
                                        preprocess=self.pre_process)
        print('Epoch {:05}, Iteration {:09}, Regular FID {}'.format(
            self.current_epoch, self.current_iteration, regular_fid_value))
        if self.cfg.trainer.model_average:
            avg_net_G_for_evaluation = \
                functools.partial(self.net_G.module.averaged_model,
                                  random_style=True)
            fid_path = self._get_save_path('average_fid', 'npy')
            fid_value = compute_fid(fid_path, self.val_data_loader,
                                    avg_net_G_for_evaluation,
                                    preprocess=self.pre_process)
            print('Epoch {:05}, Iteration {:09}, FID {}'.format(
                self.current_epoch, self.current_iteration, fid_value))
            self.net_G.float()
            return regular_fid_value, fid_value
        else:
            self.net_G.float()
            return regular_fid_value
