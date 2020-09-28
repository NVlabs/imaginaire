# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from imaginaire.discriminators.multires_patch import NLayerPatchDiscriminator
from imaginaire.model_utils.fs_vid2vid import get_fg_mask, pick_image
from imaginaire.utils.data import (get_paired_input_image_channel_number,
                                   get_paired_input_label_channel_number)
from imaginaire.utils.misc import get_nested_attr


class Discriminator(nn.Module):
    r"""Image and video discriminator constructor.

    Args:
        dis_cfg (obj): Discriminator part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, dis_cfg, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        num_input_channels = get_paired_input_label_channel_number(data_cfg)
        if num_input_channels == 0:
            num_input_channels = getattr(data_cfg, 'label_channels', 1)
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        self.num_frames_D = data_cfg.num_frames_D
        self.num_scales = get_nested_attr(dis_cfg, 'temporal.num_scales', 0)
        num_netD_input_channels = (num_input_channels + num_img_channels)
        self.use_few_shot = 'few_shot' in data_cfg.type
        if self.use_few_shot:
            num_netD_input_channels *= 2
        self.net_D = MultiPatchDiscriminator(dis_cfg.image,
                                             num_netD_input_channels)

        self.add_dis_cfg = getattr(dis_cfg, 'additional_discriminators', None)
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                add_dis_cfg = self.add_dis_cfg[name]
                num_ch = num_img_channels * (2 if self.use_few_shot else 1)
                setattr(self, 'net_D_' + name,
                        MultiPatchDiscriminator(add_dis_cfg, num_ch))

        # Temporal discriminator.
        self.num_netDT_input_channels = num_img_channels * self.num_frames_D
        for n in range(self.num_scales):
            setattr(self, 'net_DT%d' % n,
                    MultiPatchDiscriminator(dis_cfg.temporal,
                                            self.num_netDT_input_channels))
        self.has_fg = getattr(data_cfg, 'has_foreground', False)

    def forward(self, data, net_G_output, past_frames):
        r"""Discriminator forward.

        Args:
            data (dict): Input data.
            net_G_output (dict): Generator output.
            past_frames (list of tensors): Past real frames / generator outputs.
        Returns:
            (tuple):
              - output (dict): Discriminator output.
              - past_frames (list of tensors): New past frames by adding
                current outputs.
        """
        label, real_image = data['label'], data['image']
        # Only operate on the latest output frame.
        if label.dim() == 5:
            label = label[:, -1]
        if self.use_few_shot:
            # Pick only one reference image to concat with.
            ref_idx = net_G_output['ref_idx'] \
                if 'ref_idx' in net_G_output else 0
            ref_label = pick_image(data['ref_labels'], ref_idx)
            ref_image = pick_image(data['ref_images'], ref_idx)
            # Concat references with label map as discriminator input.
            label = torch.cat([label, ref_label, ref_image], dim=1)
        fake_image = net_G_output['fake_images']
        output = dict()

        # Individual frame loss.
        pred_real, pred_fake = self.discrminate_image(self.net_D, label,
                                                      real_image, fake_image)
        output['indv'] = dict()
        output['indv']['pred_real'] = pred_real
        output['indv']['pred_fake'] = pred_fake

        if 'fake_raw_images' in net_G_output and \
                net_G_output['fake_raw_images'] is not None:
            # Raw generator output loss.
            fake_raw_image = net_G_output['fake_raw_images']
            fg_mask = get_fg_mask(data['label'], self.has_fg)
            pred_real, pred_fake = self.discrminate_image(
                self.net_D, label,
                real_image * fg_mask,
                fake_raw_image * fg_mask)
            output['raw'] = dict()
            output['raw']['pred_real'] = pred_real
            output['raw']['pred_fake'] = pred_fake

        # Additional GAN loss on specific regions.
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                # Crop corresponding regions in the image according to the
                # crop function.
                add_dis_cfg = self.add_dis_cfg[name]
                file, crop_func = add_dis_cfg.crop_func.split('::')
                file = importlib.import_module(file)
                crop_func = getattr(file, crop_func)

                real_crop = crop_func(self.data_cfg, real_image, label)
                fake_crop = crop_func(self.data_cfg, fake_image, label)
                if self.use_few_shot:
                    ref_crop = crop_func(self.data_cfg, ref_image, label)
                    if ref_crop is not None:
                        real_crop = torch.cat([real_crop, ref_crop], dim=1)
                        fake_crop = torch.cat([fake_crop, ref_crop], dim=1)

                # Feed the crops to specific discriminator.
                if fake_crop is not None:
                    net_D = getattr(self, 'net_D_' + name)
                    pred_real, pred_fake = \
                        self.discrminate_image(net_D, None,
                                               real_crop, fake_crop)
                else:
                    pred_real = pred_fake = None
                output[name] = dict()
                output[name]['pred_real'] = pred_real
                output[name]['pred_fake'] = pred_fake

        # Temporal loss.
        past_frames, skipped_frames = \
            get_all_skipped_frames(past_frames, [real_image, fake_image],
                                   self.num_scales, self.num_frames_D)

        for scale in range(self.num_scales):
            real_image, fake_image = \
                [skipped_frame[scale] for skipped_frame in skipped_frames]
            pred_real, pred_fake = self.discriminate_video(real_image,
                                                           fake_image, scale)
            output['temporal_%d' % scale] = dict()
            output['temporal_%d' % scale]['pred_real'] = pred_real
            output['temporal_%d' % scale]['pred_fake'] = pred_fake

        return output, past_frames

    def discrminate_image(self, net_D, real_A, real_B, fake_B):
        r"""Discriminate individual images.

        Args:
            net_D (obj): Discriminator network.
            real_A (NxC1xHxW tensor): Input label map.
            real_B (NxC2xHxW tensor): Real image.
            fake_B (NxC2xHxW tensor): Fake image.
        Returns:
            (tuple):
              - pred_real (NxC3xH2xW2 tensor): Output of net_D for real images.
              - pred_fake (NxC3xH2xW2 tensor): Output of net_D for fake images.
        """
        if real_A is not None:
            real_AB = torch.cat([real_A, real_B], dim=1)
            fake_AB = torch.cat([real_A, fake_B], dim=1)
        else:
            real_AB, fake_AB = real_B, fake_B

        pred_real = net_D.forward(real_AB)
        pred_fake = net_D.forward(fake_AB)
        return pred_real, pred_fake

    def discriminate_video(self, real_B, fake_B, scale):
        r"""Discriminate a sequence of images.

        Args:
            real_B (NxCxHxW tensor): Real image.
            fake_B (NxCxHxW tensor): Fake image.
            scale (int): Temporal scale.
        Returns:
            (tuple):
              - pred_real (NxC2xH2xW2 tensor): Output of net_D for real images.
              - pred_fake (NxC2xH2xW2 tensor): Output of net_D for fake images.
        """
        if real_B is None:
            return None, None
        net_DT = getattr(self, 'net_DT%d' % scale)
        height, width = real_B.shape[-2:]
        real_B = real_B.view(-1, self.num_netDT_input_channels, height, width)
        fake_B = fake_B.view(-1, self.num_netDT_input_channels, height, width)

        pred_real = net_DT.forward(real_B)
        pred_fake = net_DT.forward(fake_B)
        return pred_real, pred_fake


def get_all_skipped_frames(past_frames, new_frames, t_scales, tD):
    r"""Get temporally skipped frames from the input frames.

    Args:
        past_frames (list of tensors): Past real frames / generator outputs.
        new_frames (list of tensors): Current real frame / generated output.
        t_scales (int): Temporal scale.
        tD (int): Number of frames as input to the temporal discriminator.
    Returns:
        (tuple):
          - new_past_frames (list of tensors): Past + current frames.
          - skipped_frames (list of tensors): Temporally skipped frames using
            the given t_scales.
    """
    new_past_frames, skipped_frames = [], []
    for past_frame, new_frame in zip(past_frames, new_frames):
        skipped_frame = None
        if t_scales > 0:
            past_frame, skipped_frame = \
                get_skipped_frames(past_frame, new_frame.unsqueeze(1),
                                   t_scales, tD)
        new_past_frames.append(past_frame)
        skipped_frames.append(skipped_frame)
    return new_past_frames, skipped_frames


def get_skipped_frames(all_frames, frame, t_scales, tD):
    r"""Get temporally skipped frames from the input frames.

    Args:
        all_frames (NxTxCxHxW tensor): All past frames.
        frame (Nx1xCxHxW tensor): Current frame.
        t_scales (int): Temporal scale.
        tD (int): Number of frames as input to the temporal discriminator.
    Returns:
        (tuple):
          - all_frames (NxTxCxHxW tensor): Past + current frames.
          - skipped_frames (list of NxTxCxHxW tensors): Temporally skipped
            frames.
    """
    all_frames = torch.cat([all_frames.detach(), frame], dim=1) \
        if all_frames is not None else frame
    skipped_frames = [None] * t_scales
    for s in range(t_scales):
        # Number of skipped frames between neighboring frames (e.g. 1, 3, 9,...)
        t_step = tD ** s
        # Number of frames the final triplet frames span before skipping
        # (e.g., 2, 6, 18, ...).
        t_span = t_step * (tD-1)
        if all_frames.size(1) > t_span:
            skipped_frames[s] = all_frames[:, -(t_span+1)::t_step].contiguous()

    # Maximum number of past frames we need to keep track of.
    max_num_prev_frames = (tD ** (t_scales-1)) * (tD-1)
    # Remove past frames that are older than this number.
    if all_frames.size()[1] > max_num_prev_frames:
        all_frames = all_frames[:, -max_num_prev_frames:]
    return all_frames, skipped_frames


class MultiPatchDiscriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        dis_cfg (obj): Discriminator part of the yaml config file.
        num_input_channels (int): Number of input channels.
    """

    def __init__(self, dis_cfg, num_input_channels):
        super(MultiPatchDiscriminator, self).__init__()
        kernel_size = getattr(dis_cfg, 'kernel_size', 4)
        num_filters = getattr(dis_cfg, 'num_filters', 64)
        max_num_filters = getattr(dis_cfg, 'max_num_filters', 512)
        num_discriminators = getattr(dis_cfg, 'num_discriminators', 3)
        num_layers = getattr(dis_cfg, 'num_layers', 3)
        activation_norm_type = getattr(dis_cfg, 'activation_norm_type', 'none')
        weight_norm_type = getattr(dis_cfg, 'weight_norm_type',
                                   'spectral_norm')
        self.nets_discriminator = []
        for i in range(num_discriminators):
            net_discriminator = NLayerPatchDiscriminator(
                kernel_size,
                num_input_channels,
                num_filters,
                num_layers,
                max_num_filters,
                activation_norm_type,
                weight_norm_type)
            self.add_module('discriminator_%d' % i, net_discriminator)
            self.nets_discriminator.append(net_discriminator)

    def forward(self, input_x):
        r"""Multi-resolution patch discriminator forward.

        Args:
            input_x (N x C x H x W tensor) : Concatenation of images and
                semantic representations.
        Returns:
            (dict):
              - output (list): list of output tensors produced by individual
                patch discriminators.
              - features (list): list of lists of features produced by
                individual patch discriminators.
        """
        output_list = []
        features_list = []
        input_downsampled = input_x
        for name, net_discriminator in self.named_children():
            if not name.startswith('discriminator_'):
                continue
            output, features = net_discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = F.interpolate(
                input_downsampled, scale_factor=0.5, mode='bilinear',
                align_corners=True, recompute_scale_factor=True)
        output_x = dict()
        output_x['output'] = output_list
        output_x['features'] = features_list
        return output_x
