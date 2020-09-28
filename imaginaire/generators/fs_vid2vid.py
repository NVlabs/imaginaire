# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imaginaire.layers import (Conv2dBlock, HyperConv2dBlock, HyperRes2dBlock,
                               LinearBlock, Res2dBlock)
from imaginaire.model_utils.fs_vid2vid import (extract_valid_pose_labels,
                                               pick_image, resample)
from imaginaire.utils.data import (get_paired_input_image_channel_number,
                                   get_paired_input_label_channel_number)
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.init_weight import weights_init
from imaginaire.utils.misc import get_and_setattr, get_nested_attr


class Generator(nn.Module):
    r"""Few-shot vid2vid generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.gen_cfg = gen_cfg
        self.data_cfg = data_cfg
        self.num_frames_G = data_cfg.num_frames_G
        self.flow_cfg = flow_cfg = gen_cfg.flow

        # For pose dataset.
        self.is_pose_data = hasattr(data_cfg, 'for_pose_dataset')
        if self.is_pose_data:
            pose_cfg = data_cfg.for_pose_dataset
            self.pose_type = getattr(pose_cfg, 'pose_type', 'both')
            self.remove_face_labels = getattr(pose_cfg, 'remove_face_labels',
                                              False)

        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        self.num_downsamples = num_downsamples = \
            get_and_setattr(gen_cfg, 'num_downsamples', 5)
        conv_kernel_size = get_and_setattr(gen_cfg, 'kernel_size', 3)
        num_filters = get_and_setattr(gen_cfg, 'num_filters', 32)

        max_num_filters = getattr(gen_cfg, 'max_num_filters', 1024)
        self.max_num_filters = gen_cfg.max_num_filters = \
            min(max_num_filters, num_filters * (2 ** num_downsamples))
        # Get number of filters at each layer in the main branch.
        num_filters_each_layer = [min(self.max_num_filters,
                                      num_filters * (2 ** i))
                                  for i in range(num_downsamples + 2)]

        # Hyper normalization / convolution.
        hyper_cfg = gen_cfg.hyper
        # Use adaptive weight generation for SPADE.
        self.use_hyper_spade = hyper_cfg.is_hyper_spade
        # Use adaptive for convolutional layers in the main branch.
        self.use_hyper_conv = hyper_cfg.is_hyper_conv
        # Number of hyper layers.
        self.num_hyper_layers = getattr(hyper_cfg, 'num_hyper_layers', 4)
        if self.num_hyper_layers == -1:
            self.num_hyper_layers = num_downsamples
        gen_cfg.hyper.num_hyper_layers = self.num_hyper_layers
        # Network weight generator.
        self.weight_generator = WeightGenerator(gen_cfg, data_cfg)

        # Number of layers to perform multi-spade combine.
        self.num_multi_spade_layers = getattr(flow_cfg.multi_spade_combine,
                                              'num_layers', 3)
        # Whether to generate raw output for additional losses.
        self.generate_raw_output = getattr(flow_cfg, 'generate_raw_output',
                                           False)

        # Main branch image generation.
        padding = conv_kernel_size // 2
        activation_norm_type = get_and_setattr(gen_cfg, 'activation_norm_type',
                                               'sync_batch')
        weight_norm_type = get_and_setattr(gen_cfg, 'weight_norm_type',
                                           'spectral')
        activation_norm_params = get_and_setattr(gen_cfg,
                                                 'activation_norm_params',
                                                 None)
        spade_in_channels = []  # Input channel size in SPADE module.
        for i in range(num_downsamples + 1):
            spade_in_channels += [[num_filters_each_layer[i]]] \
                if i >= self.num_multi_spade_layers \
                else [[num_filters_each_layer[i]] * 3]

        order = getattr(gen_cfg.hyper, 'hyper_block_order', 'NAC')
        for i in reversed(range(num_downsamples + 1)):
            activation_norm_params.cond_dims = spade_in_channels[i]
            is_hyper_conv = self.use_hyper_conv and i < self.num_hyper_layers
            is_hyper_norm = self.use_hyper_spade and i < self.num_hyper_layers
            setattr(self, 'up_%d' % i, HyperRes2dBlock(
                num_filters_each_layer[i + 1], num_filters_each_layer[i],
                conv_kernel_size, padding=padding,
                weight_norm_type=weight_norm_type,
                activation_norm_type=activation_norm_type,
                activation_norm_params=activation_norm_params,
                order=order * 2,
                is_hyper_conv=is_hyper_conv, is_hyper_norm=is_hyper_norm))

        self.conv_img = Conv2dBlock(num_filters, num_img_channels,
                                    conv_kernel_size, padding=padding,
                                    nonlinearity='leakyrelu', order='AC')
        self.upsample = partial(F.interpolate, scale_factor=2)

        # Flow estimation module.
        # Whether to warp reference image and combine with the synthesized.
        self.warp_ref = getattr(flow_cfg, 'warp_ref', True)
        if self.warp_ref:
            self.flow_network_ref = FlowGenerator(flow_cfg, data_cfg, 2)
            self.ref_image_embedding = \
                LabelEmbedder(flow_cfg.multi_spade_combine.embed,
                              num_img_channels + 1)
        # At beginning of training, only train an image generator.
        self.temporal_initialized = False
        if getattr(gen_cfg, 'init_temporal', True):
            self.init_temporal_network()

    def forward(self, data):
        r"""few-shot vid2vid generator forward.

        Args:
            data (dict) : Dictionary of input data.
        Returns:
            output (dict) : Dictionary of output data.
        """
        label = data['label']
        ref_labels, ref_images = data['ref_labels'], data['ref_images']
        prev_labels, prev_images = data['prev_labels'], data['prev_images']
        is_first_frame = prev_labels is None

        if self.is_pose_data:
            label, prev_labels = extract_valid_pose_labels(
                [label, prev_labels], self.pose_type, self.remove_face_labels)
            ref_labels = extract_valid_pose_labels(
                ref_labels, self.pose_type, self.remove_face_labels,
                do_remove=False)

        # Weight generation.
        x, encoded_label, conv_weights, norm_weights, atn, atn_vis, ref_idx = \
            self.weight_generator(ref_images, ref_labels, label, is_first_frame)

        # Flow estimation.
        flow, flow_mask, img_warp, cond_inputs = \
            self.flow_generation(label, ref_labels, ref_images,
                                 prev_labels, prev_images, ref_idx)

        for i in range(len(encoded_label)):
            encoded_label[i] = [encoded_label[i]]
        if self.generate_raw_output:
            encoded_label_raw = [encoded_label[i] for i in
                                 range(self.num_multi_spade_layers)]
            x_raw = None
        encoded_label = self.SPADE_combine(encoded_label, cond_inputs)

        # Main branch image generation.
        for i in range(self.num_downsamples, -1, -1):
            conv_weight = norm_weight = [None] * 3
            if self.use_hyper_conv and i < self.num_hyper_layers:
                conv_weight = conv_weights[i]
            if self.use_hyper_spade and i < self.num_hyper_layers:
                norm_weight = norm_weights[i]

            # Main branch residual blocks.
            x = self.one_up_conv_layer(x, encoded_label,
                                       conv_weight, norm_weight, i)

            # For raw output generation.
            if self.generate_raw_output and i < self.num_multi_spade_layers:
                x_raw = self.one_up_conv_layer(x_raw, encoded_label_raw,
                                               conv_weight, norm_weight, i)
            else:
                x_raw = x

        # Final conv layer.
        if self.generate_raw_output:
            img_raw = torch.tanh(self.conv_img(x_raw))
        else:
            img_raw = None
        img_final = torch.tanh(self.conv_img(x))

        output = dict()
        output['fake_images'] = img_final
        output['fake_flow_maps'] = flow
        output['fake_occlusion_masks'] = flow_mask
        output['fake_raw_images'] = img_raw
        output['warped_images'] = img_warp
        output['attention_visualization'] = atn_vis
        output['ref_idx'] = ref_idx
        return output

    def one_up_conv_layer(self, x, encoded_label, conv_weight, norm_weight, i):
        r"""One residual block layer in the main branch.

        Args:
            x (4D tensor) : Current feature map.
            encoded_label (list of tensors) : Encoded input label maps.
            conv_weight (list of tensors) : Hyper conv weights.
            norm_weight (list of tensors) : Hyper norm weights.
            i (int) : Layer index.
        Returns:
            x (4D tensor) : Output feature map.
        """
        layer = getattr(self, 'up_' + str(i))
        x = layer(x, *encoded_label[i], conv_weights=conv_weight,
                  norm_weights=norm_weight)
        if i != 0:
            x = self.upsample(x)
        return x

    def init_temporal_network(self, cfg_init=None):
        r"""When starting training multiple frames, initialize the flow network.

        Args:
            cfg_init (dict) : Weight initialization config.
        """
        flow_cfg = self.flow_cfg
        emb_cfg = flow_cfg.multi_spade_combine.embed
        num_frames_G = self.num_frames_G
        self.temporal_initialized = True

        self.sep_prev_flownet = flow_cfg.sep_prev_flow or (num_frames_G != 2) \
            or not flow_cfg.warp_ref
        if self.sep_prev_flownet:
            self.flow_network_temp = FlowGenerator(flow_cfg, self.data_cfg,
                                                   num_frames_G)
            if cfg_init is not None:
                self.flow_network_temp.apply(weights_init(cfg_init.type,
                                                          cfg_init.gain))
        else:
            self.flow_network_temp = self.flow_network_ref

        self.sep_prev_embedding = emb_cfg.sep_warp_embed or \
            not flow_cfg.warp_ref
        if self.sep_prev_embedding:
            num_img_channels = get_paired_input_image_channel_number(
                self.data_cfg)
            self.prev_image_embedding = \
                LabelEmbedder(emb_cfg, num_img_channels + 1)
            if cfg_init is not None:
                self.prev_image_embedding.apply(
                    weights_init(cfg_init.type, cfg_init.gain))
        else:
            self.prev_image_embedding = self.ref_image_embedding

        if self.warp_ref:
            if self.sep_prev_flownet:
                self.init_network_weights(self.flow_network_ref,
                                          self.flow_network_temp)
                print('Initialized temporal flow network with the reference '
                      'one.')
            if self.sep_prev_embedding:
                self.init_network_weights(self.ref_image_embedding,
                                          self.prev_image_embedding)
                print('Initialized temporal embedding network with the '
                      'reference one.')
            self.flow_temp_is_initalized = True

    def init_network_weights(self, net_src, net_dst):
        r"""Initialize weights in net_dst with those in net_src."""
        source_weights = net_src.state_dict()
        target_weights = net_dst.state_dict()

        for k, v in source_weights.items():
            if k in target_weights and target_weights[k].size() == v.size():
                target_weights[k] = v
        net_dst.load_state_dict(target_weights)

    def load_pretrained_network(self, pretrained_dict, prefix='module.'):
        r"""Load the pretrained network into self network.

        Args:
            pretrained_dict (dict): Pretrained network weights.
            prefix (str): Prefix to the network weights name.
        """
        # print(pretrained_dict.keys())
        model_dict = self.state_dict()
        print('Pretrained network has fewer layers; The following are '
              'not initialized:')

        not_initialized = set()
        for k, v in model_dict.items():
            kp = prefix + k
            if kp in pretrained_dict and v.size() == pretrained_dict[kp].size():
                model_dict[k] = pretrained_dict[kp]
            else:
                not_initialized.add('.'.join(k.split('.')[:2]))
        print(sorted(not_initialized))
        self.load_state_dict(model_dict)

    def reset(self):
        r"""Reset the network at the beginning of a sequence."""
        self.weight_generator.reset()

    def flow_generation(self, label, ref_labels, ref_images, prev_labels,
                        prev_images, ref_idx):
        r"""Generates flows and masks for warping reference / previous images.

        Args:
            label (NxCxHxW tensor): Target label map.
            ref_labels (NxKxCxHxW tensor): Reference label maps.
            ref_images (NxKx3xHxW tensor): Reference images.
            prev_labels (NxTxCxHxW tensor): Previous label maps.
            prev_images (NxTx3xHxW tensor): Previous images.
            ref_idx (Nx1 tensor): Index for which image to use from the
            reference images.
        Returns:
            (tuple):
              - flow (list of Nx2xHxW tensor): Optical flows.
              - occ_mask (list of Nx1xHxW tensor): Occlusion masks.
              - img_warp (list of Nx3xHxW tensor): Warped reference / previous
                images.
              - cond_inputs (list of Nx4xHxW tensor): Conditional inputs for
                SPADE combination.
        """
        # Pick an image in the reference images using ref_idx.
        ref_label, ref_image = pick_image([ref_labels, ref_images], ref_idx)
        # Only start using prev frames when enough prev frames are generated.
        has_prev = prev_labels is not None and \
            prev_labels.shape[1] == (self.num_frames_G - 1)
        flow, occ_mask, img_warp, cond_inputs = [None] * 2, [None] * 2, \
                                                [None] * 2, [None] * 2
        if self.warp_ref:
            # Generate flows/masks for warping the reference image.
            flow_ref, occ_mask_ref = \
                self.flow_network_ref(label, ref_label, ref_image)
            ref_image_warp = resample(ref_image, flow_ref)
            flow[0], occ_mask[0], img_warp[0] = \
                flow_ref, occ_mask_ref, ref_image_warp[:, :3]
            # Concat warped image and occlusion mask to form the conditional
            # input.
            cond_inputs[0] = torch.cat([img_warp[0], occ_mask[0]], dim=1)

        if self.temporal_initialized and has_prev:
            # Generate flows/masks for warping the previous image.
            b, t, c, h, w = prev_labels.shape
            prev_labels_concat = prev_labels.view(b, -1, h, w)
            prev_images_concat = prev_images.view(b, -1, h, w)
            flow_prev, occ_mask_prev = \
                self.flow_network_temp(label, prev_labels_concat,
                                       prev_images_concat)
            img_prev_warp = resample(prev_images[:, -1], flow_prev)
            flow[1], occ_mask[1], img_warp[1] = \
                flow_prev, occ_mask_prev, img_prev_warp
            cond_inputs[1] = torch.cat([img_warp[1], occ_mask[1]], dim=1)

        return flow, occ_mask, img_warp, cond_inputs

    def SPADE_combine(self, encoded_label, cond_inputs):
        r"""Using Multi-SPADE to combine raw synthesized image with warped
        images.

        Args:
            encoded_label (list of tensors): Original label map embeddings.
            cond_inputs (list of tensors): New SPADE conditional inputs from the
                warped images.
        Returns:
            encoded_label (list of tensors): Combined conditional inputs.
        """
        # Generate the conditional embeddings from inputs.
        embedded_img_feat = [None, None]
        if cond_inputs[0] is not None:
            embedded_img_feat[0] = self.ref_image_embedding(cond_inputs[0])
        if cond_inputs[1] is not None:
            embedded_img_feat[1] = self.prev_image_embedding(cond_inputs[1])

        # Combine the original encoded label maps with new conditional
        # embeddings.
        for i in range(self.num_multi_spade_layers):
            encoded_label[i] += [w[i] if w is not None else None
                                 for w in embedded_img_feat]
        return encoded_label

    def custom_init(self):
        r"""This function is for dealing with the numerical issue that might
        occur when doing mixed precision training.
        """
        print('Use custom initialization for the generator.')
        for k, m in self.named_modules():
            if 'weight_generator.ref_label_' in k and 'norm' in k:
                m.eps = 1e-1


class WeightGenerator(nn.Module):
    r"""Weight generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.embed_cfg = embed_cfg = gen_cfg.embed
        self.embed_arch = embed_cfg.arch

        num_filters = gen_cfg.num_filters
        self.max_num_filters = gen_cfg.max_num_filters
        self.num_downsamples = num_downsamples = gen_cfg.num_downsamples
        self.num_filters_each_layer = num_filters_each_layer = \
            [min(self.max_num_filters, num_filters * (2 ** i))
             for i in range(num_downsamples + 2)]
        if getattr(embed_cfg, 'num_filters', 32) != num_filters:
            raise ValueError('Embedding network must have the same number of '
                             'filters as generator.')

        # Normalization params.
        hyper_cfg = gen_cfg.hyper
        kernel_size = getattr(hyper_cfg, 'kernel_size', 3)
        activation_norm_type = getattr(hyper_cfg, 'activation_norm_type',
                                       'sync_batch')
        weight_norm_type = getattr(hyper_cfg, 'weight_norm_type', 'spectral')
        # Conv kernel size in main branch.
        self.conv_kernel_size = conv_kernel_size = gen_cfg.kernel_size
        # Conv kernel size in embedding network.
        self.embed_kernel_size = embed_kernel_size = \
            getattr(gen_cfg.embed, 'kernel_size', 3)
        # Conv kernel size in SPADE.
        self.kernel_size = kernel_size = \
            getattr(gen_cfg.activation_norm_params, 'kernel_size', 1)
        # Input channel size in SPADE module.
        self.spade_in_channels = []
        for i in range(num_downsamples + 1):
            self.spade_in_channels += [num_filters_each_layer[i]]

        # Hyper normalization / convolution.
        # Use adaptive weight generation for SPADE.
        self.use_hyper_spade = hyper_cfg.is_hyper_spade
        # Use adaptive for the label embedding network.
        self.use_hyper_embed = hyper_cfg.is_hyper_embed
        # Use adaptive for convolutional layers in the main branch.
        self.use_hyper_conv = hyper_cfg.is_hyper_conv
        # Number of hyper layers.
        self.num_hyper_layers = hyper_cfg.num_hyper_layers
        # Order of operations in the conv block.
        order = getattr(gen_cfg.hyper, 'hyper_block_order', 'NAC')
        self.conv_before_norm = order.find('C') < order.find('N')

        # For reference image encoding.
        # How to utilize the reference label map: concat | mul.
        self.concat_ref_label = 'concat' in hyper_cfg.method_to_use_ref_labels
        self.mul_ref_label = 'mul' in hyper_cfg.method_to_use_ref_labels
        # Output spatial size for adaptive pooling layer.
        self.sh_fix = self.sw_fix = 32
        # Number of fc layers in weight generation.
        self.num_fc_layers = getattr(hyper_cfg, 'num_fc_layers', 2)

        # Reference image encoding network.
        num_input_channels = get_paired_input_label_channel_number(data_cfg)
        if num_input_channels == 0:
            num_input_channels = getattr(data_cfg, 'label_channels', 1)
        elif get_nested_attr(data_cfg, 'for_pose_dataset.pose_type',
                             'both') == 'open':
            num_input_channels -= 3
        data_cfg.num_input_channels = num_input_channels
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        num_ref_channels = num_img_channels + (num_input_channels
                                               if self.concat_ref_label else 0)
        conv_2d_block = partial(
            Conv2dBlock, kernel_size=kernel_size,
            padding=(kernel_size // 2), weight_norm_type=weight_norm_type,
            activation_norm_type=activation_norm_type,
            nonlinearity='leakyrelu')

        self.ref_img_first = conv_2d_block(num_ref_channels, num_filters)
        if self.mul_ref_label:
            self.ref_label_first = conv_2d_block(num_input_channels,
                                                 num_filters)

        for i in range(num_downsamples):
            in_ch, out_ch = num_filters_each_layer[i], \
                num_filters_each_layer[i + 1]
            setattr(self, 'ref_img_down_%d' % i,
                    conv_2d_block(in_ch, out_ch, stride=2))
            setattr(self, 'ref_img_up_%d' % i, conv_2d_block(out_ch, in_ch))
            if self.mul_ref_label:
                setattr(self, 'ref_label_down_%d' % i,
                        conv_2d_block(in_ch, out_ch, stride=2))
                setattr(self, 'ref_label_up_%d' % i,
                        conv_2d_block(out_ch, in_ch))

        # Normalization / main branch conv weight generation.
        if self.use_hyper_spade or self.use_hyper_conv:
            for i in range(self.num_hyper_layers):
                ch_in, ch_out = num_filters_each_layer[i], \
                    num_filters_each_layer[i + 1]
                conv_ks2 = conv_kernel_size ** 2
                embed_ks2 = embed_kernel_size ** 2
                spade_ks2 = kernel_size ** 2
                spade_in_ch = self.spade_in_channels[i]

                fc_names, fc_ins, fc_outs = [], [], []
                if self.use_hyper_spade:
                    fc0_out = fcs_out = (spade_in_ch * spade_ks2 + 1) * (
                        1 if self.conv_before_norm else 2)
                    fc1_out = (spade_in_ch * spade_ks2 + 1) * (
                        1 if ch_in != ch_out else 2)
                    fc_names += ['fc_spade_0', 'fc_spade_1', 'fc_spade_s']
                    fc_ins += [ch_out] * 3
                    fc_outs += [fc0_out, fc1_out, fcs_out]
                    if self.use_hyper_embed:
                        fc_names += ['fc_spade_e']
                        fc_ins += [ch_out]
                        fc_outs += [ch_in * embed_ks2 + 1]
                if self.use_hyper_conv:
                    fc0_out = ch_out * conv_ks2 + 1
                    fc1_out = ch_in * conv_ks2 + 1
                    fcs_out = ch_out + 1
                    fc_names += ['fc_conv_0', 'fc_conv_1', 'fc_conv_s']
                    fc_ins += [ch_in] * 3
                    fc_outs += [fc0_out, fc1_out, fcs_out]

                linear_block = partial(LinearBlock,
                                       weight_norm_type='spectral',
                                       nonlinearity='leakyrelu')
                for n, l in enumerate(fc_names):
                    fc_in = fc_ins[n] if self.mul_ref_label \
                        else self.sh_fix * self.sw_fix
                    fc_layer = [linear_block(fc_in, ch_out)]
                    for k in range(1, self.num_fc_layers):
                        fc_layer += [linear_block(ch_out, ch_out)]
                    fc_layer += [LinearBlock(ch_out, fc_outs[n],
                                             weight_norm_type='spectral')]
                    setattr(self, '%s_%d' % (l, i), nn.Sequential(*fc_layer))

        # Label embedding network.
        num_hyper_layers = self.num_hyper_layers if self.use_hyper_embed else 0
        self.label_embedding = LabelEmbedder(self.embed_cfg,
                                             num_input_channels,
                                             num_hyper_layers=num_hyper_layers)

        # For multiple reference images.
        if hasattr(hyper_cfg, 'attention'):
            self.num_downsample_atn = get_and_setattr(hyper_cfg.attention,
                                                      'num_downsamples', 2)
            if data_cfg.initial_few_shot_K > 1:
                self.attention_module = AttentionModule(hyper_cfg, data_cfg,
                                                        conv_2d_block,
                                                        num_filters_each_layer)
        else:
            self.num_downsample_atn = 0

    def forward(self, ref_image, ref_label, label, is_first_frame):
        r"""Generate network weights based on the reference images.

        Args:
            ref_image (NxKx3xHxW tensor): Reference images.
            ref_label (NxKxCxHxW tensor): Reference labels.
            label (NxCxHxW tensor): Target label.
            is_first_frame (bool): Whether the current frame is the first frame.

        Returns:
            (tuple):
              - x (NxC2xH2xW2 tensor): Encoded features from reference images
                for the main branch (as input to the decoder).
              - encoded_label (list of tensors): Encoded target label map for
                SPADE.
              - conv_weights (list of tensors): Network weights for conv
                layers in the main network.
              - norm_weights (list of tensors): Network weights for SPADE
                layers in the main network.
              - attention (Nx(KxH1xW1)x(H1xW1) tensor): Attention maps.
              - atn_vis (1x1xH1xW1 tensor): Visualization for attention
                scores.
              - ref_idx (Nx1 tensor): Index for which image to use from the
                reference images.
        """
        b, k, c, h, w = ref_image.size()
        ref_image = ref_image.view(b * k, -1, h, w)
        if ref_label is not None:
            ref_label = ref_label.view(b * k, -1, h, w)

        # Encode the reference images to get the features.
        x, encoded_ref, atn, atn_vis, ref_idx = \
            self.encode_reference(ref_image, ref_label, label, k)

        # If the reference image has changed, recompute the network weights.
        if self.training or is_first_frame or k > 1:
            embedding_weights, norm_weights, conv_weights = [], [], []
            for i in range(self.num_hyper_layers):
                if self.use_hyper_spade:
                    feat = encoded_ref[min(len(encoded_ref) - 1, i + 1)]
                    embedding_weight, norm_weight = \
                        self.get_norm_weights(feat, i)
                    embedding_weights.append(embedding_weight)
                    norm_weights.append(norm_weight)
                if self.use_hyper_conv:
                    feat = encoded_ref[min(len(encoded_ref) - 1, i)]
                    conv_weights.append(self.get_conv_weights(feat, i))

            if not self.training:
                self.embedding_weights, self.conv_weights, self.norm_weights \
                    = embedding_weights, conv_weights, norm_weights
        else:
            # print('Reusing network weights.')
            embedding_weights, conv_weights, norm_weights \
                = self.embedding_weights, self.conv_weights, self.norm_weights

        # Encode the target label to get the encoded features.
        encoded_label = self.label_embedding(label, weights=(
            embedding_weights if self.use_hyper_embed else None))

        return x, encoded_label, conv_weights, norm_weights, \
            atn, atn_vis, ref_idx

    def encode_reference(self, ref_image, ref_label, label, k):
        r"""Encode the reference image to get features for weight generation.

        Args:
            ref_image ((NxK)x3xHxW tensor): Reference images.
            ref_label ((NxK)xCxHxW tensor): Reference labels.
            label (NxCxHxW tensor): Target label.
            k (int): Number of reference images.
        Returns:
            (tuple):
              - x (NxC2xH2xW2 tensor): Encoded features from reference images
                for the main branch (as input to the decoder).
              - encoded_ref (list of tensors): Encoded features from reference
                images for the weight generation branch.
              - attention (Nx(KxH1xW1)x(H1xW1) tensor): Attention maps.
              - atn_vis (1x1xH1xW1 tensor): Visualization for attention scores.
              - ref_idx (Nx1 tensor): Index for which image to use from the
                reference images.
        """
        if self.concat_ref_label:
            # Concat reference label map and image together for encoding.
            concat_ref = torch.cat([ref_image, ref_label], dim=1)
            x = self.ref_img_first(concat_ref)
        elif self.mul_ref_label:
            # Apply conv to both reference label and image, then multiply them
            # together for encoding.
            x = self.ref_img_first(ref_image)
            x_label = self.ref_label_first(ref_label)
        else:
            x = self.ref_img_first(ref_image)

        # Attention map and the index of the most similar reference image.
        atn = atn_vis = ref_idx = None
        for i in range(self.num_downsamples):
            x = getattr(self, 'ref_img_down_' + str(i))(x)
            if self.mul_ref_label:
                x_label = getattr(self, 'ref_label_down_' + str(i))(x_label)

            # Combine different reference images at a particular layer.
            if k > 1 and i == self.num_downsample_atn - 1:
                x, atn, atn_vis = self.attention_module(x, label, ref_label)
                if self.mul_ref_label:
                    x_label, _, _ = self.attention_module(x_label, None, None,
                                                          atn)

                atn_sum = atn.view(label.shape[0], k, -1).sum(2)
                ref_idx = torch.argmax(atn_sum, dim=1)

        # Get all corresponding layers in the encoder output for generating
        # weights in corresponding layers.
        encoded_image_ref = [x]
        if self.mul_ref_label:
            encoded_ref_label = [x_label]

        for i in reversed(range(self.num_downsamples)):
            conv = getattr(self, 'ref_img_up_' + str(i))(
                encoded_image_ref[-1])
            encoded_image_ref.append(conv)
            if self.mul_ref_label:
                conv_label = getattr(self, 'ref_label_up_' + str(i))(
                    encoded_ref_label[-1])
                encoded_ref_label.append(conv_label)

        if self.mul_ref_label:
            encoded_ref = []
            for i in range(len(encoded_image_ref)):
                conv, conv_label \
                    = encoded_image_ref[i], encoded_ref_label[i]
                b, c, h, w = conv.size()
                conv_label = nn.Softmax(dim=1)(conv_label)
                conv_prod = (conv.view(b, c, 1, h * w) *
                             conv_label.view(b, 1, c,
                                             h * w)).sum(3, keepdim=True)
                encoded_ref.append(conv_prod)
        else:
            encoded_ref = encoded_image_ref
        encoded_ref = encoded_ref[::-1]

        return x, encoded_ref, atn, atn_vis, ref_idx

    def get_norm_weights(self, x, i):
        r"""Adaptively generate weights for SPADE in layer i of generator.

        Args:
            x (NxCxHxW tensor): Input features.
            i (int): Layer index.
        Returns:
            (tuple):
              - embedding_weights (list of tensors): Weights for the label
                embedding network.
              - norm_weights (list of tensors): Weights for the SPADE layers.
        """
        if not self.mul_ref_label:
            # Get fixed output size for fc layers.
            x = nn.AdaptiveAvgPool2d((self.sh_fix, self.sw_fix))(x)

        in_ch = self.num_filters_each_layer[i]
        out_ch = self.num_filters_each_layer[i + 1]
        spade_ch = self.spade_in_channels[i]
        eks, sks = self.embed_kernel_size, self.kernel_size

        b = x.size(0)
        weight_reshaper = WeightReshaper()
        x = weight_reshaper.reshape_embed_input(x)

        # Weights for the label embedding network.
        embedding_weights = None
        if self.use_hyper_embed:
            fc_e = getattr(self, 'fc_spade_e_' + str(i))(x).view(b, -1)
            if 'decoder' in self.embed_arch:
                weight_shape = [in_ch, out_ch, eks, eks]
                fc_e = fc_e[:, :-in_ch]
            else:
                weight_shape = [out_ch, in_ch, eks, eks]
            embedding_weights = weight_reshaper.reshape_weight(fc_e,
                                                               weight_shape)

        # Weights for the 3 layers in SPADE module: conv_0, conv_1,
        # and shortcut.
        fc_0 = getattr(self, 'fc_spade_0_' + str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_spade_1_' + str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_spade_s_' + str(i))(x).view(b, -1)
        if self.conv_before_norm:
            out_ch = in_ch
        weight_0 = weight_reshaper.reshape_weight(fc_0, [out_ch * 2, spade_ch,
                                                         sks, sks])
        weight_1 = weight_reshaper.reshape_weight(fc_1, [in_ch * 2, spade_ch,
                                                         sks, sks])
        weight_s = weight_reshaper.reshape_weight(fc_s, [out_ch * 2, spade_ch,
                                                         sks, sks])
        norm_weights = [weight_0, weight_1, weight_s]

        return embedding_weights, norm_weights

    def get_conv_weights(self, x, i):
        r"""Adaptively generate weights for layer i in main branch convolutions.

        Args:
            x (NxCxHxW tensor): Input features.
            i (int): Layer index.
        Returns:
            (tuple):
              - conv_weights (list of tensors): Weights for the conv layers in
                the main branch.
        """
        if not self.mul_ref_label:
            x = nn.AdaptiveAvgPool2d((self.sh_fix, self.sw_fix))(x)
        in_ch = self.num_filters_each_layer[i]
        out_ch = self.num_filters_each_layer[i + 1]
        cks = self.conv_kernel_size
        b = x.size()[0]
        weight_reshaper = WeightReshaper()
        x = weight_reshaper.reshape_embed_input(x)

        fc_0 = getattr(self, 'fc_conv_0_' + str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_conv_1_' + str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_conv_s_' + str(i))(x).view(b, -1)
        weight_0 = weight_reshaper.reshape_weight(fc_0, [in_ch, out_ch,
                                                         cks, cks])
        weight_1 = weight_reshaper.reshape_weight(fc_1, [in_ch, in_ch,
                                                         cks, cks])
        weight_s = weight_reshaper.reshape_weight(fc_s, [in_ch, out_ch, 1, 1])
        return [weight_0, weight_1, weight_s]

    def reset(self):
        r"""Reset the network at the beginning of a sequence."""
        self.embedding_weights = self.conv_weights = self.norm_weights = None


class WeightReshaper():
    r"""Handles all weight reshape related tasks."""
    def reshape_weight(self, x, weight_shape):
        r"""Reshape input x to the desired weight shape.

        Args:
            x (tensor or list of tensors): Input features.
            weight_shape (list of int): Desired shape of the weight.
        Returns:
            (tuple):
              - weight (tensor): Network weights
              - bias (tensor): Network bias.
        """
        # If desired shape is a list, first divide x into the target list of
        # features.
        if type(weight_shape[0]) == list and type(x) != list:
            x = self.split_weights(x, self.sum_mul(weight_shape))

        if type(x) == list:
            return [self.reshape_weight(xi, wi)
                    for xi, wi in zip(x, weight_shape)]

        # Get output shape, and divide x into either weight + bias or
        # just weight.
        weight_shape = [x.size(0)] + weight_shape
        bias_size = weight_shape[1]
        try:
            weight = x[:, :-bias_size].view(weight_shape)
            bias = x[:, -bias_size:]
        except Exception:
            weight = x.view(weight_shape)
            bias = None
        return [weight, bias]

    def split_weights(self, weight, sizes):
        r"""When the desired shape is a list, first divide the input to each
        corresponding weight shape in the list.

        Args:
            weight (tensor): Input weight.
            sizes (int or list of int): Target sizes.
        Returns:
            weight (list of tensors): Divided weights.
        """
        if isinstance(sizes, list):
            weights = []
            cur_size = 0
            for i in range(len(sizes)):
                # For each target size in sizes, get the number of elements
                # needed.
                next_size = cur_size + self.sum(sizes[i])
                # Recursively divide the weights.
                weights.append(self.split_weights(
                    weight[:, cur_size:next_size], sizes[i]))
                cur_size = next_size
            assert (next_size == weight.size(1))
            return weights
        return weight

    def reshape_embed_input(self, x):
        r"""Reshape input to be (B x C) X H X W.

        Args:
            x (tensor or list of tensors): Input features.
        Returns:
            x (tensor or list of tensors): Reshaped features.
        """
        if isinstance(x, list):
            return [self.reshape_embed_input(xi) for xi in zip(x)]
        b, c, _, _ = x.size()
        x = x.view(b * c, -1)
        return x

    def sum(self, x):
        r"""Sum all elements recursively in a nested list.

        Args:
            x (nested list of int): Input list of elements.
        Returns:
            out (int): Sum of all elements.
        """
        if type(x) != list:
            return x
        return sum([self.sum(xi) for xi in x])

    def sum_mul(self, x):
        r"""Given a weight shape, compute the number of elements needed for
        weight + bias. If input is a list of shapes, sum all the elements.

        Args:
            x (list of int): Input list of elements.
        Returns:
            out (int or list of int): Summed number of elements.
        """
        assert (type(x) == list)
        if type(x[0]) != list:
            return np.prod(x) + x[0]  # x[0] accounts for bias.
        return [self.sum_mul(xi) for xi in x]


class AttentionModule(nn.Module):
    r"""Attention module constructor.

    Args:
       atn_cfg (obj): Generator definition part of the yaml config file.
       data_cfg (obj): Data definition part of the yaml config file
       conv_2d_block: Conv2DBlock constructor.
       num_filters_each_layer (int): The number of filters in each layer.
    """

    def __init__(self, atn_cfg, data_cfg, conv_2d_block,
                 num_filters_each_layer):
        super().__init__()
        self.initial_few_shot_K = data_cfg.initial_few_shot_K
        num_input_channels = data_cfg.num_input_channels
        num_filters = getattr(atn_cfg, 'num_filters', 32)

        self.num_downsample_atn = getattr(atn_cfg, 'num_downsamples', 2)
        self.atn_query_first = conv_2d_block(num_input_channels, num_filters)
        self.atn_key_first = conv_2d_block(num_input_channels, num_filters)
        for i in range(self.num_downsamples_atn):
            f_in, f_out = num_filters_each_layer[i], \
                num_filters_each_layer[i + 1]
            setattr(self, 'atn_key_%d' % i,
                    conv_2d_block(f_in, f_out, stride=2))
            setattr(self, 'atn_query_%d' % i,
                    conv_2d_block(f_in, f_out, stride=2))

    def forward(self, in_features, label, ref_label, attention=None):
        r"""Get the attention map to combine multiple image features in the
        case of multiple reference images.

        Args:
            in_features ((NxK)xC1xH1xW1 tensor): Input feaures.
            label (NxC2xH2xW2 tensor): Target label.
            ref_label (NxC2xH2xW2 tensor): Reference label.
            attention (Nx(KxH1xW1)x(H1xW1) tensor): Attention maps.
        Returns:
            (tuple):
              - out_features (NxC1xH1xW1 tensor): Attention-combined features.
              - attention (Nx(KxH1xW1)x(H1xW1) tensor): Attention maps.
              - atn_vis (1x1xH1xW1 tensor): Visualization for attention scores.
        """
        b, c, h, w = in_features.size()
        k = self.initial_few_shot_K
        b = b // k

        if attention is None:
            # Compute the attention map by encoding ref_label and label as
            # key and query. The map represents how much energy for the k-th
            # map at location (h_i, w_j) can contribute to the final map at
            # location (h_i2, w_j2).
            atn_key = self.attention_encode(ref_label, 'atn_key')
            atn_query = self.attention_encode(label, 'atn_query')

            atn_key = atn_key.view(b, k, c, -1).permute(
                0, 1, 3, 2).contiguous().view(b, -1, c)  # B X KHW X C
            atn_query = atn_query.view(b, c, -1)  # B X C X HW
            energy = torch.bmm(atn_key, atn_query)  # B X KHW X HW
            attention = nn.Softmax(dim=1)(energy)

        # Combine the K features from different ref images into one by using
        # the attention map.
        in_features = in_features.view(b, k, c, h * w).permute(
            0, 2, 1, 3).contiguous().view(b, c, -1)  # B X C X KHW
        out_features = torch.bmm(in_features, attention).view(b, c, h, w)

        # Get a slice of the attention map for visualization.
        atn_vis = attention.view(b, k, h * w, h * w).sum(2).view(b, k, h, w)
        return out_features, attention, atn_vis[-1:, 0:1]

    def attention_encode(self, img, net_name):
        r"""Encode the input image to get the attention map.

        Args:
            img (NxCxHxW tensor): Input image.
            net_name (str): Name for attention network.
        Returns:
            x (NxC2xH2xW2 tensor): Encoded feature.
        """
        x = getattr(self, net_name + '_first')(img)
        for i in range(self.num_downsample_atn):
            x = getattr(self, net_name + '_' + str(i))(x)
        return x


class FlowGenerator(nn.Module):
    r"""flow generator constructor.

    Args:
       flow_cfg (obj): Flow definition part of the yaml config file.
       data_cfg (obj): Data definition part of the yaml config file.
       num_frames (int): Number of input frames.
    """

    def __init__(self, flow_cfg, data_cfg, num_frames):
        super().__init__()
        num_input_channels = data_cfg.num_input_channels
        if num_input_channels == 0:
            num_input_channels = 1
        num_prev_img_channels = get_paired_input_image_channel_number(data_cfg)
        num_downsamples = getattr(flow_cfg, 'num_downsamples', 3)
        kernel_size = getattr(flow_cfg, 'kernel_size', 3)
        padding = kernel_size // 2
        num_blocks = getattr(flow_cfg, 'num_blocks', 6)
        num_filters = getattr(flow_cfg, 'num_filters', 32)
        max_num_filters = getattr(flow_cfg, 'max_num_filters', 1024)
        num_filters_each_layer = [min(max_num_filters, num_filters * (2 ** i))
                                  for i in range(num_downsamples + 1)]

        self.flow_output_multiplier = getattr(flow_cfg,
                                              'flow_output_multiplier', 20)
        self.sep_up_mask = getattr(flow_cfg, 'sep_up_mask', False)
        activation_norm_type = getattr(flow_cfg, 'activation_norm_type',
                                       'sync_batch')
        weight_norm_type = getattr(flow_cfg, 'weight_norm_type', 'spectral')

        base_conv_block = partial(Conv2dBlock, kernel_size=kernel_size,
                                  padding=padding,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  nonlinearity='leakyrelu')

        num_input_channels = num_input_channels * num_frames + \
            num_prev_img_channels * (num_frames - 1)
        # First layer.
        down_flow = [base_conv_block(num_input_channels, num_filters)]

        # Downsamples.
        for i in range(num_downsamples):
            down_flow += [base_conv_block(num_filters_each_layer[i],
                                          num_filters_each_layer[i + 1],
                                          stride=2)]

        # Resnet blocks.
        res_flow = []
        ch = num_filters_each_layer[num_downsamples]
        for i in range(num_blocks):
            res_flow += [
                Res2dBlock(ch, ch, kernel_size, padding=padding,
                           weight_norm_type=weight_norm_type,
                           activation_norm_type=activation_norm_type,
                           order='NACNAC')]

        # Upsamples.
        up_flow = []
        for i in reversed(range(num_downsamples)):
            up_flow += [nn.Upsample(scale_factor=2),
                        base_conv_block(num_filters_each_layer[i + 1],
                                        num_filters_each_layer[i])]

        conv_flow = [Conv2dBlock(num_filters, 2, kernel_size, padding=padding)]
        conv_mask = [Conv2dBlock(num_filters, 1, kernel_size, padding=padding,
                                 nonlinearity='sigmoid')]

        self.down_flow = nn.Sequential(*down_flow)
        self.res_flow = nn.Sequential(*res_flow)
        self.up_flow = nn.Sequential(*up_flow)
        if self.sep_up_mask:
            self.up_mask = nn.Sequential(*copy.deepcopy(up_flow))
        self.conv_flow = nn.Sequential(*conv_flow)
        self.conv_mask = nn.Sequential(*conv_mask)

    def forward(self, label, ref_label, ref_image):
        r"""Flow generator forward.

        Args:
            label (4D tensor) : Input label tensor.
            ref_label (4D tensor) : Reference label tensors.
            ref_image (4D tensor) : Reference image tensors.
        Returns:
            (tuple):
              - flow (4D tensor) : Generated flow map.
              - mask (4D tensor) : Generated occlusion mask.
        """
        label_concat = torch.cat([label, ref_label, ref_image], dim=1)
        downsample = self.down_flow(label_concat)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)
        flow = self.conv_flow(flow_feat) * self.flow_output_multiplier

        mask_feat = self.up_mask(res) if self.sep_up_mask else flow_feat
        mask = self.conv_mask(mask_feat)
        return flow, mask


class LabelEmbedder(nn.Module):
    r"""Embed the input label map to get embedded features.

    Args:
        emb_cfg (obj): Embed network configuration.
        num_input_channels (int): Number of input channels.
        num_hyper_layers (int): Number of hyper layers.
    """

    def __init__(self, emb_cfg, num_input_channels, num_hyper_layers=0):
        super().__init__()
        num_filters = getattr(emb_cfg, 'num_filters', 32)
        max_num_filters = getattr(emb_cfg, 'max_num_filters', 1024)
        self.arch = getattr(emb_cfg, 'arch', 'encoderdecoder')
        self.num_downsamples = num_downsamples = \
            getattr(emb_cfg, 'num_downsamples', 5)
        kernel_size = getattr(emb_cfg, 'kernel_size', 3)
        weight_norm_type = getattr(emb_cfg, 'weight_norm_type', 'spectral')
        activation_norm_type = getattr(emb_cfg, 'activation_norm_type', 'none')

        self.unet = 'unet' in self.arch
        self.has_decoder = 'decoder' in self.arch or self.unet
        self.num_hyper_layers = num_hyper_layers \
            if num_hyper_layers != -1 else num_downsamples

        base_conv_block = partial(HyperConv2dBlock, kernel_size=kernel_size,
                                  padding=(kernel_size // 2),
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  nonlinearity='leakyrelu')

        ch = [min(max_num_filters, num_filters * (2 ** i))
              for i in range(num_downsamples + 1)]

        self.conv_first = base_conv_block(num_input_channels, num_filters,
                                          activation_norm_type='none')

        # Downsample.
        for i in range(num_downsamples):
            is_hyper_conv = (i < num_hyper_layers) and not self.has_decoder
            setattr(self, 'down_%d' % i,
                    base_conv_block(ch[i], ch[i + 1], stride=2,
                                    is_hyper_conv=is_hyper_conv))

        # Upsample.
        if self.has_decoder:
            self.upsample = nn.Upsample(scale_factor=2)
            for i in reversed(range(num_downsamples)):
                ch_i = ch[i + 1] * (
                    2 if self.unet and i != num_downsamples - 1 else 1)
                setattr(self, 'up_%d' % i,
                        base_conv_block(ch_i, ch[i],
                                        is_hyper_conv=(i < num_hyper_layers)))

    def forward(self, input, weights=None):
        r"""Embedding network forward.

        Args:
            input (NxCxHxW tensor): Network input.
            weights (list of tensors): Conv weights if using hyper network.
        Returns:
            output (list of tensors): Network outputs at different layers.
        """
        if input is None:
            return None
        output = [self.conv_first(input)]

        for i in range(self.num_downsamples):
            layer = getattr(self, 'down_%d' % i)
            # For hyper networks, the hyper layers are at the last few layers
            # of decoder (if the network has a decoder). Otherwise, the hyper
            # layers will be at the first few layers of the network.
            if i >= self.num_hyper_layers or self.has_decoder:
                conv = layer(output[-1])
            else:
                conv = layer(output[-1], conv_weights=weights[i])
            # We will use outputs from different layers as input to different
            # SPADE layers in the main branch.
            output.append(conv)

        if not self.has_decoder:
            return output

        # If the network has a decoder, will use outputs from the decoder
        # layers instead of the encoding layers.
        if not self.unet:
            output = [output[-1]]

        for i in reversed(range(self.num_downsamples)):
            input_i = output[-1]
            if self.unet and i != self.num_downsamples - 1:
                input_i = torch.cat([input_i, output[i + 1]], dim=1)

            input_i = self.upsample(input_i)
            layer = getattr(self, 'up_%d' % i)
            # The last few layers will be hyper layers if necessary.
            if i >= self.num_hyper_layers:
                conv = layer(input_i)
            else:
                conv = layer(input_i, conv_weights=weights[i])
            output.append(conv)

        if self.unet:
            output = output[self.num_downsamples:]
        return output[::-1]
