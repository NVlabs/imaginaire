# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# The file is duplicated from https://github.com/NVIDIA/flownet2-pytorch
# with some modifications.
'''
Portions of this code copyright 2017, Clement Pinard
'''
from torch.nn import init
import torch
import torch.nn as nn
from .submodules import conv, predict_flow, deconv


class FlowNetS(nn.Module):
    r"""FlowNet2 S module. Check out the FlowNet2 paper for more details
    https://arxiv.org/abs/1612.01925

    Args:
        args (obj): Network initialization arguments
        input_channels (int): Number of input channels. Default is 12.
        use_batch_norm (bool): Use batch norm or not. Default is true.
    """
    def __init__(self, args, input_channels=12, use_batch_norm=True):
        super(FlowNetS, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.conv1 = conv(
            self.use_batch_norm,
            input_channels,
            64,
            kernel_size=7,
            stride=2)
        self.conv2 = conv(self.use_batch_norm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.use_batch_norm, 128, 256, kernel_size=5,
                          stride=2)
        self.conv3_1 = conv(self.use_batch_norm, 256, 256)
        self.conv4 = conv(self.use_batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.use_batch_norm, 512, 512)
        self.conv5 = conv(self.use_batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.use_batch_norm, 512, 512)
        self.conv6 = conv(self.use_batch_norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.use_batch_norm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensors of concatenated images.
        Returns:
            flow2 (tensor): Output flow tensors.
        """
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,
