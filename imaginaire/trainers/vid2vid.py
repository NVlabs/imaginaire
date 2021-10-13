# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

from torch.cuda.amp import autocast
import imageio
import numpy as np
import torch
from tqdm import tqdm

from imaginaire.evaluation.fid import compute_fid
from imaginaire.losses import (FeatureMatchingLoss, FlowLoss, GANLoss,
                               PerceptualLoss)
from imaginaire.model_utils.fs_vid2vid import (concat_frames, detach,
                                               get_fg_mask,
                                               pre_process_densepose, resample)
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import is_master
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.misc import get_nested_attr, split_labels, to_cuda
from imaginaire.utils.visualization import (tensor2flow, tensor2im, tensor2label)
from imaginaire.utils.visualization.pose import tensor2pose


class Trainer(BaseTrainer):
    r"""Initialize vid2vid trainer.

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
        # Below is for testing setting, the FID computation during training
        # is just for getting a quick idea of the performance. It does not
        # equal to the final performance evaluation.
        # Below, we will determine how many videos that we want to do
        # evaluation, and the length of each video.
        # It is better to keep the number of videos to be multiple of 8 so
        # that all the GPUs in a node will contribute equally to the
        # evaluation. None of them is idol.
        self.sample_size = (
            getattr(cfg.trainer, 'num_videos_to_test', 64),
            getattr(cfg.trainer, 'num_frames_per_video', 10)
        )

        self.sequence_length = 1
        if not self.is_inference:
            self.train_dataset = self.train_data_loader.dataset
            self.sequence_length_max = \
                min(getattr(cfg.data.train, 'max_sequence_length', 100),
                    self.train_dataset.sequence_length_max)
        self.Tensor = torch.cuda.FloatTensor
        self.has_fg = getattr(cfg.data, 'has_foreground', False)

        self.net_G_output = self.data_prev = None
        self.net_G_module = self.net_G.module
        if self.cfg.trainer.model_average_config.enabled:
            self.net_G_module = self.net_G_module.module

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
        r"""Initialize training loss terms. In vid2vid, in addition to
        the GAN loss, feature matching loss, and perceptual loss used in
        pix2pixHD, we also add temporal GAN (and feature matching) loss,
        and flow warping loss. Optionally, we can also add an additional
        face discriminator for the face region.

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

        # Perceptual loss.
        perceptual_loss = cfg.trainer.perceptual_loss
        self._assign_criteria('Perceptual',
                              PerceptualLoss(
                                  network=perceptual_loss.mode,
                                  layers=perceptual_loss.layers,
                                  weights=perceptual_loss.weights,
                                  num_scales=getattr(perceptual_loss,
                                                     'num_scales', 1)),
                              loss_weight.perceptual)

        # L1 Loss.
        if getattr(loss_weight, 'L1', 0) > 0:
            self._assign_criteria('L1', torch.nn.L1Loss(), loss_weight.L1)

        # Whether to add an additional discriminator for specific regions.
        self.add_dis_cfg = getattr(self.cfg.dis, 'additional_discriminators',
                                   None)
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                add_dis_cfg = self.add_dis_cfg[name]
                self.weights['GAN_' + name] = add_dis_cfg.loss_weight
                self.weights['FeatureMatching_' + name] = \
                    loss_weight.feature_matching

        # Temporal GAN loss.
        self.num_temporal_scales = get_nested_attr(self.cfg.dis,
                                                   'temporal.num_scales', 0)
        for s in range(self.num_temporal_scales):
            self.weights['GAN_T%d' % s] = loss_weight.temporal_gan
            self.weights['FeatureMatching_T%d' % s] = \
                loss_weight.feature_matching

        # Flow loss. It consists of three parts: L1 loss compared to GT,
        # warping loss when used to warp images, and loss on the occlusion mask.
        self.use_flow = hasattr(cfg.gen, 'flow')
        if self.use_flow:
            self.criteria['Flow'] = FlowLoss(cfg)
            self.weights['Flow'] = self.weights['Flow_L1'] = \
                self.weights['Flow_Warp'] = \
                self.weights['Flow_Mask'] = loss_weight.flow

        # Other custom losses.
        self._define_custom_losses()

    def _define_custom_losses(self):
        r"""All other custom losses are defined here."""
        pass

    def _start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch. When current_epoch is smaller than
        $(single_frame_epoch), we only train a single frame and the generator is
        just an image generator. After that, we start doing temporal training
        and train multiple frames. We will double the number of training frames
        every $(num_epochs_temporal_step) epochs.

        Args:
            current_epoch (int): Current number of epoch.
        """
        cfg = self.cfg
        # Only generates one frame at the beginning of training
        if current_epoch < cfg.single_frame_epoch:
            self.train_dataset.sequence_length = 1
        # Then add the temporal network to generator, and train multiple frames.
        elif current_epoch == cfg.single_frame_epoch:
            self.init_temporal_network()

        # Double the length of training sequence every few epochs.
        temp_epoch = current_epoch - cfg.single_frame_epoch
        if temp_epoch > 0:
            sequence_length = \
                cfg.data.train.initial_sequence_length * \
                (2 ** (temp_epoch // cfg.num_epochs_temporal_step))
            sequence_length = min(sequence_length, self.sequence_length_max)
            if sequence_length > self.sequence_length:
                self.sequence_length = sequence_length
                self.train_dataset.set_sequence_length(sequence_length)
                print('------- Updating sequence length to %d -------' %
                      sequence_length)

    def init_temporal_network(self):
        r"""Initialize temporal training when beginning to train multiple
        frames. Set the sequence length to $(initial_sequence_length).
        """
        self.tensorboard_init = False
        # Update training sequence length.
        self.sequence_length = self.cfg.data.train.initial_sequence_length
        if not self.is_inference:
            self.train_dataset.set_sequence_length(self.sequence_length)
            print('------ Now start training %d frames -------' %
                  self.sequence_length)

    def _start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current number of iteration.
        """
        data = self.pre_process(data)
        return to_cuda(data)

    def pre_process(self, data):
        r"""Do any data pre-processing here.

        Args:
            data (dict): Data used for the current iteration.
        """
        data_cfg = self.cfg.data
        if hasattr(data_cfg, 'for_pose_dataset') and \
                ('pose_maps-densepose' in data_cfg.input_labels):
            pose_cfg = data_cfg.for_pose_dataset
            data['label'] = pre_process_densepose(pose_cfg, data['label'],
                                                  self.is_inference)
        return data

    def post_process(self, data, net_G_output):
        r"""Do any postprocessing of the data / output here.

        Args:
            data (dict): Training data at the current iteration.
            net_G_output (dict): Output of the generator.
        """
        return data, net_G_output

    def gen_update(self, data):
        r"""Update the vid2vid generator. We update in the fashion of
        dis_update (frame 1), gen_update (frame 1),
        dis_update (frame 2), gen_update (frame 2), ... in each iteration.

        Args:
            data (dict): Training data at the current iteration.
        """
        # Whether to reuse generator output for both gen_update and
        # dis_update. It saves time but consumes a bit more memory.
        reuse_gen_output = getattr(self.cfg.trainer, 'reuse_gen_output', True)

        past_frames = [None, None]
        net_G_output = None
        data_prev = None
        for t in range(self.sequence_length):
            data_t = self.get_data_t(data, net_G_output, data_prev, t)
            data_prev = data_t

            # Discriminator update.
            if reuse_gen_output:
                net_G_output = self.net_G(data_t)
            else:
                with torch.no_grad():
                    net_G_output = self.net_G(data_t)
            data_t, net_G_output = self.post_process(data_t, net_G_output)

            # Get losses and update D if image generated by network in training.
            if 'fake_images_source' not in net_G_output:
                net_G_output['fake_images_source'] = 'in_training'
            if net_G_output['fake_images_source'] != 'pretrained':
                net_D_output, _ = self.net_D(data_t, detach(net_G_output), past_frames)
                self.get_dis_losses(net_D_output)

            # Generator update.
            if not reuse_gen_output:
                net_G_output = self.net_G(data_t)
                data_t, net_G_output = self.post_process(data_t, net_G_output)

            # Get losses and update G if image generated by network in training.
            if 'fake_images_source' not in net_G_output:
                net_G_output['fake_images_source'] = 'in_training'
            if net_G_output['fake_images_source'] != 'pretrained':
                net_D_output, past_frames = \
                    self.net_D(data_t, net_G_output, past_frames)
                self.get_gen_losses(data_t, net_G_output, net_D_output)

            # update average
            if self.cfg.trainer.model_average_config.enabled:
                self.net_G.module.update_average()

    def dis_update(self, data):
        r"""The update is already done in gen_update.

        Args:
            data (dict): Training data at the current iteration.
        """
        pass

    def reset(self):
        r"""Reset the trainer (for inference) at the beginning of a sequence.
        """
        # print('Resetting trainer.')
        self.net_G_output = self.data_prev = None
        self.t = 0

        self.test_in_model_average_mode = getattr(
            self, 'test_in_model_average_mode', self.cfg.trainer.model_average_config.enabled)
        if self.test_in_model_average_mode:
            net_G_module = self.net_G.module.averaged_model
        else:
            net_G_module = self.net_G.module
        if hasattr(net_G_module, 'reset'):
            net_G_module.reset()

    def create_sequence_output_dir(self, output_dir, key):
        r"""Create output subdir for this sequence.

        Args:
            output_dir (str): Root output dir.
            key (str): LMDB key which contains sequence name and file name.
        Returns:
            output_dir (str): Output subdir for this sequence.
            seq_name (str): Name of this sequence.
        """
        seq_dir = '/'.join(key.split('/')[:-1])
        output_dir = os.path.join(output_dir, seq_dir)
        os.makedirs(output_dir, exist_ok=True)
        seq_name = seq_dir.replace('/', '-')
        return output_dir, seq_name

    def test(self, test_data_loader, root_output_dir, inference_args):
        r"""Run inference on all sequences.

        Args:
            test_data_loader (object): Test data loader.
            root_output_dir (str): Location to dump outputs.
            inference_args (optional): Optional args.
        """

        # Go over all sequences.
        loader = test_data_loader
        num_inference_sequences = loader.dataset.num_inference_sequences()
        for sequence_idx in range(num_inference_sequences):
            loader.dataset.set_inference_sequence_idx(sequence_idx)
            print('Seq id: %d, Seq length: %d' %
                  (sequence_idx + 1, len(loader)))

            # Reset model at start of new inference sequence.
            self.reset()
            self.sequence_length = len(loader)

            # Go over all frames of this sequence.
            video = []
            for idx, data in enumerate(tqdm(loader)):
                key = data['key']['images'][0][0]
                filename = key.split('/')[-1]

                # Create output dir for this sequence.
                if idx == 0:
                    output_dir, seq_name = \
                        self.create_sequence_output_dir(root_output_dir, key)
                    video_path = os.path.join(output_dir, '..', seq_name)

                # Get output and save images.
                data['img_name'] = filename
                data = self.start_of_iteration(data, current_iteration=-1)
                output = self.test_single(data, output_dir, inference_args)
                video.append(output)

            # Save output as mp4.
            imageio.mimsave(video_path + '.mp4', video, fps=15)

    def test_single(self, data, output_dir=None, inference_args=None):
        r"""The inference function. If output_dir exists, also save the
        output image.
        Args:
            data (dict): Training data at the current iteration.
            output_dir (str): Save image directory.
            inference_args (obj): Inference args.
        """
        if getattr(inference_args, 'finetune', False):
            if not getattr(self, 'has_finetuned', False):
                self.finetune(data, inference_args)

        net_G = self.net_G
        if self.test_in_model_average_mode:
            net_G = net_G.module.averaged_model
        net_G.eval()

        data_t = self.get_data_t(data, self.net_G_output, self.data_prev, 0)
        if self.is_inference or self.sequence_length > 1:
            self.data_prev = data_t

        # Generator forward.
        with torch.no_grad():
            self.net_G_output = net_G(data_t)

        if output_dir is None:
            return self.net_G_output

        save_fake_only = getattr(inference_args, 'save_fake_only', False)
        if save_fake_only:
            image_grid = tensor2im(self.net_G_output['fake_images'])[0]
        else:
            vis_images = self.get_test_output_images(data)
            image_grid = np.hstack([np.vstack(im) for im in
                                    vis_images if im is not None])
        if 'img_name' in data:
            save_name = data['img_name'].split('.')[0] + '.jpg'
        else:
            save_name = '%04d.jpg' % self.t
        output_filename = os.path.join(output_dir, save_name)
        os.makedirs(output_dir, exist_ok=True)
        imageio.imwrite(output_filename, image_grid)
        self.t += 1

        return image_grid

    def get_test_output_images(self, data):
        r"""Get the visualization output of test function.

        Args:
            data (dict): Training data at the current iteration.
        """
        vis_images = [
            self.visualize_label(data['label'][:, -1]),
            tensor2im(data['images'][:, -1]),
            tensor2im(self.net_G_output['fake_images']),
        ]
        return vis_images

    def gen_frames(self, data, use_model_average=False):
        r"""Generate a sequence of frames given a sequence of data.

        Args:
            data (dict): Training data at the current iteration.
            use_model_average (bool): Whether to use model average
                for update or not.
        """
        net_G_output = None  # Previous generator output.
        data_prev = None  # Previous data.
        if use_model_average:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G

        # Iterate through the length of sequence.
        all_info = {'inputs': [], 'outputs': []}
        for t in range(self.sequence_length):
            # Get the data at the current time frame.
            data_t = self.get_data_t(data, net_G_output, data_prev, t)
            data_prev = data_t

            # Generator forward.
            with torch.no_grad():
                net_G_output = net_G(data_t)

            # Do any postprocessing if necessary.
            data_t, net_G_output = self.post_process(data_t, net_G_output)

            if t == 0:
                # Get the output at beginning of sequence for visualization.
                first_net_G_output = net_G_output

            all_info['inputs'].append(data_t)
            all_info['outputs'].append(net_G_output)

        return first_net_G_output, net_G_output, all_info

    def get_gen_losses(self, data_t, net_G_output, net_D_output):
        r"""Compute generator losses.

        Args:
            data_t (dict): Training data at the current time t.
            net_G_output (dict): Output of the generator.
            net_D_output (dict): Output of the discriminator.
        """
        update_finished = False
        while not update_finished:
            with autocast(enabled=self.cfg.trainer.amp_config.enabled):
                # Individual frame GAN loss and feature matching loss.
                self.gen_losses['GAN'], self.gen_losses['FeatureMatching'] = \
                    self.compute_gan_losses(net_D_output['indv'],
                                            dis_update=False)

                # Perceptual loss.
                self.gen_losses['Perceptual'] = self.criteria['Perceptual'](
                    net_G_output['fake_images'], data_t['image'])

                # L1 loss.
                if getattr(self.cfg.trainer.loss_weight, 'L1', 0) > 0:
                    self.gen_losses['L1'] = self.criteria['L1'](
                        net_G_output['fake_images'], data_t['image'])

                # Raw (hallucinated) output image losses (GAN and perceptual).
                if 'raw' in net_D_output:
                    raw_GAN_losses = self.compute_gan_losses(
                        net_D_output['raw'], dis_update=False
                    )
                    fg_mask = get_fg_mask(data_t['label'], self.has_fg)
                    raw_perceptual_loss = self.criteria['Perceptual'](
                        net_G_output['fake_raw_images'] * fg_mask,
                        data_t['image'] * fg_mask)
                    self.gen_losses['GAN'] += raw_GAN_losses[0]
                    self.gen_losses['FeatureMatching'] += raw_GAN_losses[1]
                    self.gen_losses['Perceptual'] += raw_perceptual_loss

                # Additional discriminator losses.
                if self.add_dis_cfg is not None:
                    for name in self.add_dis_cfg:
                        (self.gen_losses['GAN_' + name],
                         self.gen_losses['FeatureMatching_' + name]) = \
                            self.compute_gan_losses(net_D_output[name],
                                                    dis_update=False)

                # Flow and mask loss.
                if self.use_flow:
                    (self.gen_losses['Flow_L1'], self.gen_losses['Flow_Warp'],
                     self.gen_losses['Flow_Mask']) = self.criteria['Flow'](
                        data_t, net_G_output, self.current_epoch)

                # Temporal GAN loss and feature matching loss.
                if self.cfg.trainer.loss_weight.temporal_gan > 0:
                    if self.sequence_length > 1:
                        for s in range(self.num_temporal_scales):
                            loss_GAN, loss_FM = self.compute_gan_losses(
                                net_D_output['temporal_%d' % s],
                                dis_update=False
                            )
                            self.gen_losses['GAN_T%d' % s] = loss_GAN
                            self.gen_losses['FeatureMatching_T%d' % s] = loss_FM

                # Other custom losses.
                self._get_custom_gen_losses(data_t, net_G_output, net_D_output)

                # Sum all losses together.
                total_loss = self.Tensor(1).fill_(0)
                for key in self.gen_losses:
                    if key != 'total':
                        total_loss += self.gen_losses[key] * self.weights[key]
                self.gen_losses['total'] = total_loss

            # Zero-grad and backpropagate the loss.
            self.opt_G.zero_grad(set_to_none=True)
            self.scaler_G.scale(total_loss).backward()

            # Optionally clip gradient norm.
            if hasattr(self.cfg.gen_opt, 'clip_grad_norm'):
                self.scaler_G.unscale_(self.opt_G)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net_G_module.parameters(),
                    self.cfg.gen_opt.clip_grad_norm
                )
                if torch.isfinite(total_norm) and \
                        total_norm > self.cfg.gen_opt.clip_grad_norm:
                    print(f"Gradient norm of the generator ({total_norm}) "
                          f"too large, clipping it to "
                          f"{self.cfg.gen_opt.clip_grad_norm}.")

            # Perform an optimizer step.
            self.scaler_G.step(self.opt_G)
            self.scaler_G.update()
            # Whether the step above was skipped.
            if self.last_step_count_G == self.opt_G._step_count:
                print("Generator overflowed!")
            else:
                self.last_step_count_G = self.opt_G._step_count
                update_finished = True

    def _get_custom_gen_losses(self, data_t, net_G_output, net_D_output):
        r"""All other custom generator losses go here.

        Args:
            data_t (dict): Training data at the current time t.
            net_G_output (dict): Output of the generator.
            net_D_output (dict): Output of the discriminator.
        """
        pass

    def get_dis_losses(self, net_D_output):
        r"""Compute discriminator losses.

        Args:
            net_D_output (dict): Output of the discriminator.
        """
        update_finished = False
        while not update_finished:
            with autocast(enabled=self.cfg.trainer.amp_config.enabled):
                # Individual frame GAN loss.
                self.dis_losses['GAN'] = self.compute_gan_losses(
                    net_D_output['indv'], dis_update=True
                )

                # Raw (hallucinated) output image GAN loss.
                if 'raw' in net_D_output:
                    raw_loss = self.compute_gan_losses(net_D_output['raw'],
                                                       dis_update=True)
                    self.dis_losses['GAN'] += raw_loss

                # Additional GAN loss.
                if self.add_dis_cfg is not None:
                    for name in self.add_dis_cfg:
                        self.dis_losses['GAN_' + name] = \
                            self.compute_gan_losses(net_D_output[name],
                                                    dis_update=True)

                # Temporal GAN loss.
                if self.cfg.trainer.loss_weight.temporal_gan > 0:
                    if self.sequence_length > 1:
                        for s in range(self.num_temporal_scales):
                            self.dis_losses['GAN_T%d' % s] = \
                                self.compute_gan_losses(
                                    net_D_output['temporal_%d' % s],
                                    dis_update=True
                                )

                # Other custom losses.
                self._get_custom_dis_losses(net_D_output)

                # Sum all losses together.
                total_loss = self.Tensor(1).fill_(0)
                for key in self.dis_losses:
                    if key != 'total':
                        total_loss += self.dis_losses[key] * self.weights[key]
                self.dis_losses['total'] = total_loss

            # Zero-grad and backpropagate the loss.
            self.opt_D.zero_grad(set_to_none=True)
            self._time_before_backward()
            self.scaler_D.scale(total_loss).backward()

            # Optionally clip gradient norm.
            if hasattr(self.cfg.dis_opt, 'clip_grad_norm'):
                self.scaler_D.unscale_(self.opt_D)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net_D.parameters(), self.cfg.dis_opt.clip_grad_norm
                )
                if torch.isfinite(total_norm) and \
                        total_norm > self.cfg.dis_opt.clip_grad_norm:
                    print(f"Gradient norm of the discriminator ({total_norm}) "
                          f"too large, clipping it to "
                          f"{self.cfg.dis_opt.clip_grad_norm}.")

            # Perform an optimizer step.
            self._time_before_step()
            self.scaler_D.step(self.opt_D)
            self.scaler_D.update()
            # Whether the step above was skipped.
            if self.last_step_count_D == self.opt_D._step_count:
                print("Discriminator overflowed!")
            else:
                self.last_step_count_D = self.opt_D._step_count
                update_finished = True

    def _get_custom_dis_losses(self, net_D_output):
        r"""All other custom losses go here.

        Args:
            net_D_output (dict): Output of the discriminator.
        """
        pass

    def compute_gan_losses(self, net_D_output, dis_update):
        r"""Compute GAN loss and feature matching loss.

        Args:
            net_D_output (dict): Output of the discriminator.
            dis_update (bool): Whether to update discriminator.
        """
        if net_D_output['pred_fake'] is None:
            return self.Tensor(1).fill_(0) if dis_update else [
                self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)]
        if dis_update:
            # Get the GAN loss for real/fake outputs.
            GAN_loss = \
                self.criteria['GAN'](net_D_output['pred_fake']['output'], False,
                                     dis_update=True) + \
                self.criteria['GAN'](net_D_output['pred_real']['output'], True,
                                     dis_update=True)
            return GAN_loss
        else:
            # Get the GAN loss and feature matching loss for fake output.
            GAN_loss = self.criteria['GAN'](
                net_D_output['pred_fake']['output'], True, dis_update=False)

            FM_loss = self.criteria['FeatureMatching'](
                net_D_output['pred_fake']['features'],
                net_D_output['pred_real']['features'])
            return GAN_loss, FM_loss

    def get_data_t(self, data, net_G_output, data_prev, t):
        r"""Get data at current time frame given the sequence of data.

        Args:
            data (dict): Training data for current iteration.
            net_G_output (dict): Output of the generator (for previous frame).
            data_prev (dict): Data for previous frame.
            t (int): Current time.
        """
        label = data['label'][:, t]
        image = data['images'][:, t]

        if data_prev is not None:
            # Concat previous labels/fake images to the ones before.
            num_frames_G = self.cfg.data.num_frames_G
            prev_labels = concat_frames(data_prev['prev_labels'],
                                        data_prev['label'], num_frames_G - 1)
            prev_images = concat_frames(
                data_prev['prev_images'],
                net_G_output['fake_images'].detach(), num_frames_G - 1)
        else:
            prev_labels = prev_images = None

        data_t = dict()
        data_t['label'] = label
        data_t['image'] = image
        data_t['prev_labels'] = prev_labels
        data_t['prev_images'] = prev_images
        data_t['real_prev_image'] = data['images'][:, t - 1] if t > 0 else None
        return data_t

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Print the errors to console."""
        if not torch.distributed.is_initialized():
            if current_iteration % self.cfg.logging_iter == 0:
                message = '(epoch: %d, iters: %d) ' % (current_epoch,
                                                       current_iteration)
                for k, v in self.gen_losses.items():
                    if k != 'total':
                        message += '%s: %.3f,  ' % (k, v)
                message += '\n'
                for k, v in self.dis_losses.items():
                    if k != 'total':
                        message += '%s: %.3f,  ' % (k, v)
                print(message)

    def write_metrics(self):
        r"""If moving average model presents, we have two meters one for
        regular FID and one for average FID. If no moving average model,
        we just report average FID.
        """
        if self.cfg.trainer.model_average_config.enabled:
            regular_fid, average_fid = self._compute_fid()
            if regular_fid is None or average_fid is None:
                return
            metric_dict = {'FID/average': average_fid, 'FID/regular': regular_fid}
            self._write_to_meters(metric_dict, self.metric_meters, reduce=False)
        else:
            regular_fid = self._compute_fid()
            if regular_fid is None:
                return
            metric_dict = {'FID/regular': regular_fid}
            self._write_to_meters(metric_dict, self.metric_meters, reduce=False)
        self._flush_meters(self.metric_meters)

    def _compute_fid(self):
        r"""Compute FID values."""
        self.net_G.eval()
        self.net_G_output = None
        # Due to complicated video evaluation procedure we are using, we will
        # pass the trainer to the evaluation code instead of the
        # generator network.
        # net_G_for_evaluation = self.net_G
        trainer = self
        self.test_in_model_average_mode = False
        regular_fid_path = self._get_save_path('regular_fid', 'npy')
        few_shot = True if 'few_shot' in self.cfg.data.type else False
        regular_fid_value = compute_fid(regular_fid_path, self.val_data_loader,
                                        trainer,
                                        sample_size=self.sample_size,
                                        is_video=True, few_shot_video=few_shot)
        print('Epoch {:05}, Iteration {:09}, Regular FID {}'.format(
            self.current_epoch, self.current_iteration, regular_fid_value))
        if self.cfg.trainer.model_average_config.enabled:
            # Due to complicated video evaluation procedure we are using,
            # we will pass the trainer to the evaluation code instead of the
            # generator network.
            # avg_net_G_for_evaluation = self.net_G.module.averaged_model
            trainer_avg_mode = self
            self.test_in_model_average_mode = True
            # The above flag will be reset after computing FID.
            fid_path = self._get_save_path('average_fid', 'npy')
            few_shot = True if 'few_shot' in self.cfg.data.type else False
            fid_value = compute_fid(fid_path, self.val_data_loader,
                                    trainer_avg_mode,
                                    sample_size=self.sample_size,
                                    is_video=True, few_shot_video=few_shot)
            print('Epoch {:05}, Iteration {:09}, Average FID {}'.format(
                self.current_epoch, self.current_iteration, fid_value))
            self.net_G.float()
            return regular_fid_value, fid_value
        else:
            self.net_G.float()
            return regular_fid_value

    def visualize_label(self, label):
        r"""Visualize the input label when saving to image.

        Args:
            label (tensor): Input label tensor.
        """
        cfgdata = self.cfg.data
        if hasattr(cfgdata, 'for_pose_dataset'):
            label = tensor2pose(self.cfg, label)
        elif hasattr(cfgdata, 'input_labels') and \
                'seg_maps' in cfgdata.input_labels:
            for input_type in cfgdata.input_types:
                if 'seg_maps' in input_type:
                    num_labels = cfgdata.one_hot_num_classes.seg_maps
            label = tensor2label(label, num_labels)
        elif getattr(cfgdata, 'label_channels', 1) > 3:
            label = tensor2im(label.sum(1, keepdim=True))
        else:
            label = tensor2im(label)
        return label

    def save_image(self, path, data):
        r"""Save the output images to path.
        Note when the generate_raw_output is FALSE. Then,
        first_net_G_output['fake_raw_images'] is None and will not be displayed.
        In model average mode, we will plot the flow visualization twice.
        Args:
            path (str): Save path.
            data (dict): Training data for current iteration.
        """
        self.net_G.eval()
        if self.cfg.trainer.model_average_config.enabled:
            self.net_G.module.averaged_model.eval()
        self.net_G_output = None
        with torch.no_grad():
            first_net_G_output, net_G_output, all_info = self.gen_frames(data)
            if self.cfg.trainer.model_average_config.enabled:
                first_net_G_output_avg, net_G_output_avg, _ = self.gen_frames(
                    data, use_model_average=True)

        # Visualize labels.
        label_lengths = self.train_data_loader.dataset.get_label_lengths()
        labels = split_labels(data['label'], label_lengths)
        vis_labels_start, vis_labels_end = [], []
        for key, value in labels.items():
            if key == 'seg_maps':
                vis_labels_start.append(self.visualize_label(value[:, -1]))
                vis_labels_end.append(self.visualize_label(value[:, 0]))
            else:
                vis_labels_start.append(tensor2im(value[:, -1]))
                vis_labels_end.append(tensor2im(value[:, 0]))

        if is_master():
            vis_images = [
                *vis_labels_start,
                tensor2im(data['images'][:, -1]),
                tensor2im(net_G_output['fake_images']),
                tensor2im(net_G_output['fake_raw_images'])]
            if self.cfg.trainer.model_average_config.enabled:
                vis_images += [
                    tensor2im(net_G_output_avg['fake_images']),
                    tensor2im(net_G_output_avg['fake_raw_images'])]

            if self.sequence_length > 1:
                vis_images_first = [
                    *vis_labels_end,
                    tensor2im(data['images'][:, 0]),
                    tensor2im(first_net_G_output['fake_images']),
                    tensor2im(first_net_G_output['fake_raw_images'])
                ]
                if self.cfg.trainer.model_average_config.enabled:
                    vis_images_first += [
                        tensor2im(first_net_G_output_avg['fake_images']),
                        tensor2im(first_net_G_output_avg['fake_raw_images'])]

                if self.use_flow:
                    flow_gt, conf_gt = self.criteria['Flow'].flowNet(
                        data['images'][:, -1], data['images'][:, -2])
                    warped_image_gt = resample(data['images'][:, -1], flow_gt)
                    vis_images_first += [
                        tensor2flow(flow_gt),
                        tensor2im(conf_gt, normalize=False),
                        tensor2im(warped_image_gt),
                    ]
                    vis_images += [
                        tensor2flow(net_G_output['fake_flow_maps']),
                        tensor2im(net_G_output['fake_occlusion_masks'],
                                  normalize=False),
                        tensor2im(net_G_output['warped_images']),
                    ]
                    if self.cfg.trainer.model_average_config.enabled:
                        vis_images_first += [
                            tensor2flow(flow_gt),
                            tensor2im(conf_gt, normalize=False),
                            tensor2im(warped_image_gt),
                        ]
                        vis_images += [
                            tensor2flow(net_G_output_avg['fake_flow_maps']),
                            tensor2im(net_G_output_avg['fake_occlusion_masks'],
                                      normalize=False),
                            tensor2im(net_G_output_avg['warped_images'])]

                vis_images = [[np.vstack((im_first, im))
                               for im_first, im in zip(imgs_first, imgs)]
                              for imgs_first, imgs in zip(vis_images_first,
                                                          vis_images)
                              if imgs is not None]

            image_grid = np.hstack([np.vstack(im) for im in
                                    vis_images if im is not None])

            print('Save output images to {}'.format(path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            imageio.imwrite(path, image_grid)

            # Gather all outputs for dumping into video.
            if self.sequence_length > 1:
                output_images = []
                for item in all_info['outputs']:
                    output_images.append(tensor2im(item['fake_images'])[0])

                imageio.mimwrite(os.path.splitext(path)[0] + '.mp4',
                                 output_images, fps=2, macro_block_size=None)

        self.net_G.float()
