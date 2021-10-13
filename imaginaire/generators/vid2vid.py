# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imaginaire.generators.fs_vid2vid import LabelEmbedder
from imaginaire.layers import Conv2dBlock, LinearBlock, Res2dBlock
from imaginaire.model_utils.fs_vid2vid import (extract_valid_pose_labels,
                                               resample)
from imaginaire.utils.data import (get_paired_input_image_channel_number,
                                   get_paired_input_label_channel_number)
from imaginaire.utils.init_weight import weights_init


class BaseNetwork(nn.Module):
    r"""vid2vid generator."""

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def get_num_filters(self, num_downsamples):
        r"""Get the number of filters at current layer.

        Args:
            num_downsamples (int) : How many downsamples at current layer.
        Returns:
            output (int) : Number of filters.
        """
        return min(self.max_num_filters,
                   self.num_filters * (2 ** num_downsamples))


class Generator(BaseNetwork):
    r"""vid2vid generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.gen_cfg = gen_cfg
        self.data_cfg = data_cfg
        self.num_frames_G = data_cfg.num_frames_G
        # Number of residual blocks in generator.
        self.num_layers = num_layers = getattr(gen_cfg, 'num_layers', 7)
        # Number of downsamplings for previous frame.
        self.num_downsamples_img = getattr(gen_cfg, 'num_downsamples_img', 4)
        # Number of filters in the first layer.
        self.num_filters = num_filters = getattr(gen_cfg, 'num_filters', 32)
        self.max_num_filters = getattr(gen_cfg, 'max_num_filters', 1024)
        self.kernel_size = kernel_size = getattr(gen_cfg, 'kernel_size', 3)
        padding = kernel_size // 2

        # For pose dataset.
        self.is_pose_data = hasattr(data_cfg, 'for_pose_dataset')
        if self.is_pose_data:
            pose_cfg = data_cfg.for_pose_dataset
            self.pose_type = getattr(pose_cfg, 'pose_type', 'both')
            self.remove_face_labels = getattr(pose_cfg, 'remove_face_labels',
                                              False)

        # Input data params.
        num_input_channels = get_paired_input_label_channel_number(data_cfg)
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        aug_cfg = data_cfg.val.augmentations
        if hasattr(aug_cfg, 'center_crop_h_w'):
            crop_h_w = aug_cfg.center_crop_h_w
        elif hasattr(aug_cfg, 'resize_h_w'):
            crop_h_w = aug_cfg.resize_h_w
        else:
            raise ValueError('Need to specify output size.')
        crop_h, crop_w = crop_h_w.split(',')
        crop_h, crop_w = int(crop_h), int(crop_w)
        # Spatial size at the bottle neck of generator.
        self.sh = crop_h // (2 ** num_layers)
        self.sw = crop_w // (2 ** num_layers)

        # Noise vector dimension.
        self.z_dim = getattr(gen_cfg, 'style_dims', 256)
        self.use_segmap_as_input = \
            getattr(gen_cfg, 'use_segmap_as_input', False)

        # Label / image embedding network.
        self.emb_cfg = emb_cfg = getattr(gen_cfg, 'embed', None)
        self.use_embed = getattr(emb_cfg, 'use_embed', 'True')
        self.num_downsamples_embed = getattr(emb_cfg, 'num_downsamples', 5)
        if self.use_embed:
            self.label_embedding = LabelEmbedder(emb_cfg, num_input_channels)

        # Flow network.
        self.flow_cfg = flow_cfg = gen_cfg.flow
        # Use SPADE to combine warped and hallucinated frames instead of
        # linear combination.
        self.spade_combine = getattr(flow_cfg, 'multi_spade_combine', True)
        # Number of layers to perform multi-spade combine.
        self.num_multi_spade_layers = getattr(flow_cfg.multi_spade_combine,
                                              'num_layers', 3)
        # At beginning of training, only train an image generator.
        self.temporal_initialized = False
        # Whether to output hallucinated frame (when training temporal network)
        # for additional loss.
        self.generate_raw_output = False

        # Image generation network.
        weight_norm_type = getattr(gen_cfg, 'weight_norm_type', 'spectral')
        activation_norm_type = gen_cfg.activation_norm_type
        activation_norm_params = gen_cfg.activation_norm_params
        if self.use_embed and \
                not hasattr(activation_norm_params, 'num_filters'):
            activation_norm_params.num_filters = 0
        nonlinearity = 'leakyrelu'

        self.base_res_block = base_res_block = partial(
            Res2dBlock, kernel_size=kernel_size, padding=padding,
            weight_norm_type=weight_norm_type,
            activation_norm_type=activation_norm_type,
            activation_norm_params=activation_norm_params,
            nonlinearity=nonlinearity, order='NACNAC')

        # Upsampling residual blocks.
        for i in range(num_layers, -1, -1):
            activation_norm_params.cond_dims = self.get_cond_dims(i)
            activation_norm_params.partial = self.get_partial(
                i) if hasattr(self, 'get_partial') else False
            layer = base_res_block(self.get_num_filters(i + 1),
                                   self.get_num_filters(i))
            setattr(self, 'up_%d' % i, layer)

        # Final conv layer.
        self.conv_img = Conv2dBlock(num_filters, num_img_channels,
                                    kernel_size, padding=padding,
                                    nonlinearity=nonlinearity, order='AC')

        num_filters = min(self.max_num_filters,
                          num_filters * (2 ** (self.num_layers + 1)))
        if self.use_segmap_as_input:
            self.fc = Conv2dBlock(num_input_channels, num_filters,
                                  kernel_size=3, padding=1)
        else:
            self.fc = LinearBlock(self.z_dim, num_filters * self.sh * self.sw)

        # Misc.
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = partial(F.interpolate, scale_factor=2)
        self.init_temporal_network()

    def forward(self, data):
        r"""vid2vid generator forward.

        Args:
           data (dict) : Dictionary of input data.
        Returns:
           output (dict) : Dictionary of output data.
        """
        label = data['label']
        label_prev, img_prev = data['prev_labels'], data['prev_images']
        is_first_frame = img_prev is None
        z = getattr(data, 'z', None)
        bs, _, h, w = label.size()

        if self.is_pose_data:
            label, label_prev = extract_valid_pose_labels(
                [label, label_prev], self.pose_type, self.remove_face_labels)

        # Get SPADE conditional maps by embedding current label input.
        cond_maps_now = self.get_cond_maps(label, self.label_embedding)

        # Input to the generator will either be noise/segmentation map (for
        # first frame) or encoded previous frame (for subsequent frames).
        if is_first_frame:
            # First frame in the sequence, start from scratch.
            if self.use_segmap_as_input:
                x_img = F.interpolate(label, size=(self.sh, self.sw))
                x_img = self.fc(x_img)
            else:
                if z is None:
                    z = torch.randn(bs, self.z_dim, dtype=label.dtype,
                                    device=label.get_device()).fill_(0)
                x_img = self.fc(z).view(bs, -1, self.sh, self.sw)

            # Upsampling layers.
            for i in range(self.num_layers, self.num_downsamples_img, -1):
                j = min(self.num_downsamples_embed, i)
                x_img = getattr(self, 'up_' + str(i))(x_img, *cond_maps_now[j])
                x_img = self.upsample(x_img)
        else:
            # Not the first frame, will encode the previous frame and feed to
            # the generator.
            x_img = self.down_first(img_prev[:, -1])

            # Get label embedding for the previous frame.
            cond_maps_prev = self.get_cond_maps(label_prev[:, -1],
                                                self.label_embedding)

            # Downsampling layers.
            for i in range(self.num_downsamples_img + 1):
                j = min(self.num_downsamples_embed, i)
                x_img = getattr(self, 'down_' + str(i))(x_img,
                                                        *cond_maps_prev[j])
                if i != self.num_downsamples_img:
                    x_img = self.downsample(x_img)

            # Resnet blocks.
            j = min(self.num_downsamples_embed, self.num_downsamples_img + 1)
            for i in range(self.num_res_blocks):
                cond_maps = cond_maps_prev[j] if i < self.num_res_blocks // 2 \
                    else cond_maps_now[j]
                x_img = getattr(self, 'res_' + str(i))(x_img, *cond_maps)

        flow = mask = img_warp = None

        num_frames_G = self.num_frames_G
        # Whether to warp the previous frame or not.
        warp_prev = self.temporal_initialized and not is_first_frame and \
            label_prev.shape[1] == num_frames_G - 1
        if warp_prev:
            # Estimate flow & mask.
            label_concat = torch.cat([label_prev.view(bs, -1, h, w),
                                      label], dim=1)
            img_prev_concat = img_prev.view(bs, -1, h, w)
            flow, mask = self.flow_network_temp(label_concat, img_prev_concat)
            img_warp = resample(img_prev[:, -1], flow)
            if self.spade_combine:
                # if using SPADE combine, integrate the warped image (and
                # occlusion mask) into conditional inputs for SPADE.
                img_embed = torch.cat([img_warp, mask], dim=1)
                cond_maps_img = self.get_cond_maps(img_embed,
                                                   self.img_prev_embedding)
                x_raw_img = None

        # Main image generation branch.
        for i in range(self.num_downsamples_img, -1, -1):
            # Get SPADE conditional inputs.
            j = min(i, self.num_downsamples_embed)
            cond_maps = cond_maps_now[j]

            # For raw output generation.
            if self.generate_raw_output:
                if i >= self.num_multi_spade_layers - 1:
                    x_raw_img = x_img
                if i < self.num_multi_spade_layers:
                    x_raw_img = self.one_up_conv_layer(x_raw_img, cond_maps, i)

            # For final output.
            if warp_prev and i < self.num_multi_spade_layers:
                cond_maps += cond_maps_img[j]
            x_img = self.one_up_conv_layer(x_img, cond_maps, i)

        # Final conv layer.
        img_final = torch.tanh(self.conv_img(x_img))

        img_raw = None
        if self.spade_combine and self.generate_raw_output:
            img_raw = torch.tanh(self.conv_img(x_raw_img))
        if warp_prev and not self.spade_combine:
            img_raw = img_final
            img_final = img_final * mask + img_warp * (1 - mask)

        output = dict()
        output['fake_images'] = img_final
        output['fake_flow_maps'] = flow
        output['fake_occlusion_masks'] = mask
        output['fake_raw_images'] = img_raw
        output['warped_images'] = img_warp
        return output

    def one_up_conv_layer(self, x, encoded_label, i):
        r"""One residual block layer in the main branch.

        Args:
           x (4D tensor) : Current feature map.
           encoded_label (list of tensors) : Encoded input label maps.
           i (int) : Layer index.
        Returns:
           x (4D tensor) : Output feature map.
        """
        layer = getattr(self, 'up_' + str(i))
        x = layer(x, *encoded_label)
        if i != 0:
            x = self.upsample(x)
        return x

    def init_temporal_network(self, cfg_init=None):
        r"""When starting training multiple frames, initialize the
        downsampling network and flow network.

        Args:
            cfg_init (dict) : Weight initialization config.
        """
        # Number of image downsamplings for the previous frame.
        num_downsamples_img = self.num_downsamples_img
        # Number of residual blocks for the previous frame.
        self.num_res_blocks = int(
            np.ceil((self.num_layers - num_downsamples_img) / 2.0) * 2)

        # First conv layer.
        num_img_channels = get_paired_input_image_channel_number(self.data_cfg)
        self.down_first = \
            Conv2dBlock(num_img_channels,
                        self.num_filters, self.kernel_size,
                        padding=self.kernel_size // 2)
        if cfg_init is not None:
            self.down_first.apply(weights_init(cfg_init.type, cfg_init.gain))

        # Downsampling residual blocks.
        activation_norm_params = self.gen_cfg.activation_norm_params
        for i in range(num_downsamples_img + 1):
            activation_norm_params.cond_dims = self.get_cond_dims(i)
            layer = self.base_res_block(self.get_num_filters(i),
                                        self.get_num_filters(i + 1))
            if cfg_init is not None:
                layer.apply(weights_init(cfg_init.type, cfg_init.gain))
            setattr(self, 'down_%d' % i, layer)

        # Additional residual blocks.
        res_ch = self.get_num_filters(num_downsamples_img + 1)
        activation_norm_params.cond_dims = \
            self.get_cond_dims(num_downsamples_img + 1)
        for i in range(self.num_res_blocks):
            layer = self.base_res_block(res_ch, res_ch)
            if cfg_init is not None:
                layer.apply(weights_init(cfg_init.type, cfg_init.gain))
            setattr(self, 'res_%d' % i, layer)

        # Flow network.
        flow_cfg = self.flow_cfg
        self.temporal_initialized = True
        self.generate_raw_output = getattr(flow_cfg, 'generate_raw_output',
                                           False) and self.spade_combine
        self.flow_network_temp = FlowGenerator(flow_cfg, self.data_cfg)
        if cfg_init is not None:
            self.flow_network_temp.apply(weights_init(cfg_init.type,
                                                      cfg_init.gain))

        self.spade_combine = getattr(flow_cfg, 'multi_spade_combine', True)
        if self.spade_combine:
            emb_cfg = flow_cfg.multi_spade_combine.embed
            num_img_channels = get_paired_input_image_channel_number(
                self.data_cfg)
            self.img_prev_embedding = LabelEmbedder(emb_cfg,
                                                    num_img_channels + 1)
            if cfg_init is not None:
                self.img_prev_embedding.apply(weights_init(cfg_init.type,
                                                           cfg_init.gain))

    def get_cond_dims(self, num_downs=0):
        r"""Get the dimensions of conditional inputs.

        Args:
           num_downs (int) : How many downsamples at current layer.
        Returns:
           ch (list) : List of dimensions.
        """
        if not self.use_embed:
            ch = [self.num_input_channels]
        else:
            num_filters = getattr(self.emb_cfg, 'num_filters', 32)
            num_downs = min(num_downs, self.num_downsamples_embed)
            ch = [min(self.max_num_filters, num_filters * (2 ** num_downs))]
            if (num_downs < self.num_multi_spade_layers):
                ch = ch * 2
        return ch

    def get_cond_maps(self, label, embedder):
        r"""Get the conditional inputs.

        Args:
           label (4D tensor) : Input label tensor.
           embedder (obj) : Embedding network.
        Returns:
           cond_maps (list) : List of conditional inputs.
        """
        if not self.use_embed:
            return [label] * (self.num_layers + 1)
        embedded_label = embedder(label)
        cond_maps = [embedded_label]
        cond_maps = [[m[i] for m in cond_maps] for i in
                     range(len(cond_maps[0]))]
        return cond_maps


class FlowGenerator(BaseNetwork):
    r"""Flow generator constructor.

    Args:
       flow_cfg (obj): Flow definition part of the yaml config file.
       data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, flow_cfg, data_cfg):
        super().__init__()
        num_input_channels = get_paired_input_label_channel_number(data_cfg)
        num_prev_img_channels = get_paired_input_image_channel_number(data_cfg)
        num_frames = data_cfg.num_frames_G  # Num. of input frames.

        self.num_filters = num_filters = getattr(flow_cfg, 'num_filters', 32)
        self.max_num_filters = getattr(flow_cfg, 'max_num_filters', 1024)
        num_downsamples = getattr(flow_cfg, 'num_downsamples', 5)
        kernel_size = getattr(flow_cfg, 'kernel_size', 3)
        padding = kernel_size // 2
        self.num_res_blocks = getattr(flow_cfg, 'num_res_blocks', 6)
        # Multiplier on the flow output.
        self.flow_output_multiplier = getattr(flow_cfg,
                                              'flow_output_multiplier', 20)

        activation_norm_type = getattr(flow_cfg, 'activation_norm_type',
                                       'sync_batch')
        weight_norm_type = getattr(flow_cfg, 'weight_norm_type', 'spectral')

        base_conv_block = partial(Conv2dBlock, kernel_size=kernel_size,
                                  padding=padding,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  nonlinearity='leakyrelu')

        # Will downsample the labels and prev frames separately, then combine.
        down_lbl = [base_conv_block(num_input_channels * num_frames,
                                    num_filters)]
        down_img = [base_conv_block(num_prev_img_channels * (num_frames - 1),
                                    num_filters)]
        for i in range(num_downsamples):
            down_lbl += [base_conv_block(self.get_num_filters(i),
                                         self.get_num_filters(i + 1),
                                         stride=2)]
            down_img += [base_conv_block(self.get_num_filters(i),
                                         self.get_num_filters(i + 1),
                                         stride=2)]

        # Resnet blocks.
        res_flow = []
        ch = self.get_num_filters(num_downsamples)
        for i in range(self.num_res_blocks):
            res_flow += [
                Res2dBlock(ch, ch, kernel_size, padding=padding,
                           weight_norm_type=weight_norm_type,
                           activation_norm_type=activation_norm_type,
                           order='CNACN')]

        # Upsample.
        up_flow = []
        for i in reversed(range(num_downsamples)):
            up_flow += [nn.Upsample(scale_factor=2),
                        base_conv_block(self.get_num_filters(i + 1),
                                        self.get_num_filters(i))]

        conv_flow = [Conv2dBlock(num_filters, 2, kernel_size, padding=padding)]
        conv_mask = [Conv2dBlock(num_filters, 1, kernel_size, padding=padding,
                                 nonlinearity='sigmoid')]

        self.down_lbl = nn.Sequential(*down_lbl)
        self.down_img = nn.Sequential(*down_img)
        self.res_flow = nn.Sequential(*res_flow)
        self.up_flow = nn.Sequential(*up_flow)
        self.conv_flow = nn.Sequential(*conv_flow)
        self.conv_mask = nn.Sequential(*conv_mask)

    def forward(self, label, img_prev):
        r"""Flow generator forward.

        Args:
           label (4D tensor) : Input label tensor.
           img_prev (4D tensor) : Previously generated image tensors.
        Returns:
            (tuple):
              - flow (4D tensor) : Generated flow map.
              - mask (4D tensor) : Generated occlusion mask.
        """
        downsample = self.down_lbl(label) + self.down_img(img_prev)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)
        flow = self.conv_flow(flow_feat) * self.flow_output_multiplier
        mask = self.conv_mask(flow_feat)
        return flow, mask
