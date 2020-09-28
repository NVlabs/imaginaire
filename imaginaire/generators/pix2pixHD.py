# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Upsample as NearestUpsample

from imaginaire.layers import Conv2dBlock, Res2dBlock
from imaginaire.utils.data import (get_paired_input_image_channel_number,
                                   get_paired_input_label_channel_number)
from imaginaire.utils.distributed import master_only_print as print


class Generator(nn.Module):
    r"""Pix2pixHD coarse-to-fine generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        # pix2pixHD has a global generator.
        global_gen_cfg = gen_cfg.global_generator
        num_filters_global = getattr(global_gen_cfg, 'num_filters', 64)
        # Optionally, it can have several local enhancers. They are useful
        # for generating high resolution images.
        local_gen_cfg = gen_cfg.local_enhancer
        self.num_local_enhancers = num_local_enhancers = \
            getattr(local_gen_cfg, 'num_enhancers', 1)
        # By default, pix2pixHD using instance normalization.
        activation_norm_type = getattr(gen_cfg, 'activation_norm_type',
                                       'instance')
        activation_norm_params = getattr(gen_cfg, 'activation_norm_params',
                                         None)
        weight_norm_type = getattr(gen_cfg, 'weight_norm_type', '')
        padding_mode = getattr(gen_cfg, 'padding_mode', 'reflect')
        base_conv_block = partial(Conv2dBlock,
                                  padding_mode=padding_mode,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  activation_norm_params=activation_norm_params,
                                  nonlinearity='relu')
        base_res_block = partial(Res2dBlock,
                                 padding_mode=padding_mode,
                                 weight_norm_type=weight_norm_type,
                                 activation_norm_type=activation_norm_type,
                                 activation_norm_params=activation_norm_params,
                                 nonlinearity='relu', order='CNACN')
        # Know what is the number of available segmentation labels.
        num_input_channels = get_paired_input_label_channel_number(data_cfg)
        self.concat_features = False
        # Check whether label input contains specific type of data (e.g.
        # instance_maps).
        self.contain_instance_map = False
        if data_cfg.input_labels[-1] == 'instance_maps':
            self.contain_instance_map = True
        # The feature encoder is only useful when the instance map is provided.
        if hasattr(gen_cfg, 'enc') and self.contain_instance_map:
            num_feat_channels = getattr(gen_cfg.enc, 'num_feat_channels', 0)
            if num_feat_channels > 0:
                num_input_channels += num_feat_channels
                self.concat_features = True
                self.encoder = Encoder(gen_cfg.enc, data_cfg)

        # Global generator model.
        global_model = GlobalGenerator(global_gen_cfg, data_cfg,
                                       num_input_channels, padding_mode,
                                       base_conv_block, base_res_block)
        if num_local_enhancers == 0:
            self.global_model = global_model
        else:
            # Get rid of the last layer.
            global_model = global_model.model
            global_model = [global_model[i]
                            for i in range(len(global_model) - 1)]
            # global_model = [global_model[i]
            #                 for i in range(len(global_model) - 2)]
            self.global_model = nn.Sequential(*global_model)

        # Local enhancer model.
        for n in range(num_local_enhancers):
            # num_filters = num_filters_global // (2 ** n)
            num_filters = num_filters_global // (2 ** (n + 1))
            output_img = (n == num_local_enhancers - 1)
            setattr(self, 'enhancer_%d' % n,
                    LocalEnhancer(local_gen_cfg, data_cfg,
                                  num_input_channels, num_filters,
                                  padding_mode, base_conv_block,
                                  base_res_block, output_img))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1],
                                       count_include_pad=False)

    def forward(self, data, random_style=False):
        r"""Coarse-to-fine generator forward.

        Args:
            data (dict) : Dictionary of input data.
            random_style (bool): Always set to false for the pix2pixHD model.
        Returns:
            output (dict) : Dictionary of output data.
        """
        label = data['label']

        output = dict()
        if self.concat_features:
            features = self.encoder(data['images'], data['instance_maps'])
            label = torch.cat([label, features], dim=1)
            output['feature_maps'] = features

        # Create input pyramid.
        input_downsampled = [label]
        for i in range(self.num_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        # Output at coarsest level.
        x = self.global_model(input_downsampled[-1])

        # Coarse-to-fine: build up one layer at a time.
        for n in range(self.num_local_enhancers):
            input_n = input_downsampled[self.num_local_enhancers - n - 1]
            enhancer = getattr(self, 'enhancer_%d' % n)
            x = enhancer(x, input_n)

        output['fake_images'] = x
        return output

    def load_pretrained_network(self, pretrained_dict):
        r"""Load a pretrained network."""
        # print(pretrained_dict.keys())
        model_dict = self.state_dict()
        print('Pretrained network has fewer layers; The following are '
              'not initialized:')

        not_initialized = set()
        for k, v in model_dict.items():
            kp = 'module.' + k.replace('global_model.', 'global_model.model.')
            if kp in pretrained_dict and v.size() == pretrained_dict[kp].size():
                model_dict[k] = pretrained_dict[kp]
            else:
                not_initialized.add('.'.join(k.split('.')[:2]))
        print(sorted(not_initialized))
        self.load_state_dict(model_dict)

    def inference(self, data, **kwargs):
        r"""Generator inference.

        Args:
            data (dict) : Dictionary of input data.
        Returns:
            fake_images (tensor): Output fake images.
            file_names (str): Data file name.
        """
        output = self.forward(data, **kwargs)
        return output['fake_images'], data['key']['seg_maps'][0]


class LocalEnhancer(nn.Module):
    r"""Local enhancer constructor. These are sub-networks that are useful
    when aiming to produce high-resolution outputs.

    Args:
        gen_cfg (obj): local generator definition part of the yaml config
        file.
        data_cfg (obj): Data definition part of the yaml config file.
        num_input_channels (int): Number of segmentation labels.
        num_filters (int): Number of filters for the first layer.
        padding_mode (str): zero | reflect | ...
        base_conv_block (obj): Conv block with preset attributes.
        base_res_block (obj): Residual block with preset attributes.
        output_img (bool): Output is image or feature map.
    """

    def __init__(self, gen_cfg, data_cfg, num_input_channels, num_filters,
                 padding_mode, base_conv_block, base_res_block,
                 output_img=False):
        super(LocalEnhancer, self).__init__()
        num_res_blocks = getattr(gen_cfg, 'num_res_blocks', 3)
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        # Downsample.
        model_downsample = \
            [base_conv_block(num_input_channels, num_filters, 7, padding=3),
             base_conv_block(num_filters, num_filters * 2, 3, stride=2,
                             padding=1)]
        # Residual blocks.
        model_upsample = []
        for i in range(num_res_blocks):
            model_upsample += [base_res_block(num_filters * 2, num_filters * 2,
                                              3, padding=1)]
        # Upsample.
        model_upsample += \
            [NearestUpsample(scale_factor=2),
             base_conv_block(num_filters * 2, num_filters, 3, padding=1)]

        # Final convolution.
        if output_img:
            model_upsample += [Conv2dBlock(num_filters, num_img_channels, 7,
                                           padding=3, padding_mode=padding_mode,
                                           nonlinearity='tanh')]

        self.model_downsample = nn.Sequential(*model_downsample)
        self.model_upsample = nn.Sequential(*model_upsample)

    def forward(self, output_coarse, input_fine):
        r"""Local enhancer forward.

        Args:
            output_coarse (4D tensor) : Coarse output from previous layer.
            input_fine (4D tensor) : Fine input from current layer.
        Returns:
            output (4D tensor) : Refined output.
        """
        output = self.model_upsample(self.model_downsample(input_fine)
                                     + output_coarse)
        return output


class GlobalGenerator(nn.Module):
    r"""Coarse generator constructor. This is the main generator in the
    pix2pixHD architecture.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
        num_input_channels (int): Number of segmentation labels.
        padding_mode (str): zero | reflect | ...
        base_conv_block (obj): Conv block with preset attributes.
        base_res_block (obj): Residual block with preset attributes.
    """

    def __init__(self, gen_cfg, data_cfg, num_input_channels, padding_mode,
                 base_conv_block, base_res_block):
        super(GlobalGenerator, self).__init__()
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        num_filters = getattr(gen_cfg, 'num_filters', 64)
        num_downsamples = getattr(gen_cfg, 'num_downsamples', 4)
        num_res_blocks = getattr(gen_cfg, 'num_res_blocks', 9)
        # First layer.
        model = [base_conv_block(num_input_channels, num_filters,
                                 kernel_size=7, padding=3)]
        # Downsample.
        for i in range(num_downsamples):
            ch = num_filters * (2 ** i)
            model += [base_conv_block(ch, ch * 2, 3, padding=1, stride=2)]
        # ResNet blocks.
        ch = num_filters * (2 ** num_downsamples)
        for i in range(num_res_blocks):
            model += [base_res_block(ch, ch, 3, padding=1)]
        # Upsample.
        num_upsamples = num_downsamples
        for i in reversed(range(num_upsamples)):
            ch = num_filters * (2 ** i)
            model += \
                [NearestUpsample(scale_factor=2),
                 base_conv_block(ch * 2, ch, 3, padding=1)]
        model += [Conv2dBlock(num_filters, num_img_channels, 7, padding=3,
                              padding_mode=padding_mode, nonlinearity='tanh')]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        r"""Coarse-to-fine generator forward.

        Args:
            input (4D tensor) : Input semantic representations.
        Returns:
            output (4D tensor) : Synthesized image by generator.
        """
        return self.model(input)


class Encoder(nn.Module):
    r"""Encoder for getting region-wise features for style control.

    Args:
        enc_cfg (obj): Encoder definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, enc_cfg, data_cfg):
        super(Encoder, self).__init__()
        label_nc = get_paired_input_label_channel_number(data_cfg)
        feat_nc = enc_cfg.num_feat_channels
        n_clusters = getattr(enc_cfg, 'num_clusters', 10)
        for i in range(label_nc):
            dummy_arr = np.zeros((n_clusters, feat_nc), dtype=np.float32)
            self.register_buffer('cluster_%d' % i,
                                 torch.tensor(dummy_arr, dtype=torch.float32))
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        self.num_feat_channels = getattr(enc_cfg, 'num_feat_channels', 3)
        num_filters = getattr(enc_cfg, 'num_filters', 64)
        num_downsamples = getattr(enc_cfg, 'num_downsamples', 4)
        weight_norm_type = getattr(enc_cfg, 'weight_norm_type', 'none')
        activation_norm_type = getattr(enc_cfg, 'activation_norm_type',
                                       'instance')
        padding_mode = getattr(enc_cfg, 'padding_mode', 'reflect')
        base_conv_block = partial(Conv2dBlock,
                                  padding_mode=padding_mode,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  nonlinearity='relu')
        model = [base_conv_block(num_img_channels, num_filters, 7, padding=3)]
        # Downsample.
        for i in range(num_downsamples):
            ch = num_filters * (2**i)
            model += [base_conv_block(ch, ch * 2, 3, stride=2, padding=1)]
        # Upsample.
        for i in reversed(range(num_downsamples)):
            ch = num_filters * (2 ** i)
            model += [NearestUpsample(scale_factor=2),
                      base_conv_block(ch * 2, ch, 3, padding=1)]

        model += [Conv2dBlock(num_filters, self.num_feat_channels, 7,
                              padding=3, padding_mode=padding_mode,
                              nonlinearity='tanh')]
        self.model = nn.Sequential(*model)

    def forward(self, input, instance_map):
        r"""Extracting region-wise features

        Args:
            input (4D tensor): Real RGB images.
            instance_map (4D tensor): Instance label mask.
        Returns:
            outputs_mean (4D tensor): Instance-wise average-pooled
                feature maps.
        """
        outputs = self.model(input)
        # Instance-wise average pooling.
        outputs_mean = torch.zeros_like(outputs)
        # Find all the unique labels in this batch.
        inst_list = np.unique(instance_map.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size(0)):
                # Find the pixels in this instance map have this instance label.
                indices = (instance_map[b:b+1] == int(i)).nonzero()  # n x 4
                # Scan through the feature channels.
                for j in range(self.num_feat_channels):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j,
                                         indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j,
                                 indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean
