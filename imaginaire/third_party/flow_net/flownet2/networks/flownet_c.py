# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# The file is duplicated from https://github.com/NVIDIA/flownet2-pytorch
# with some modifications.
from torch.nn import init
import correlation
import torch
import torch.nn as nn
from .submodules import conv, predict_flow, deconv, tofp16, tofp32


class FlowNetC(nn.Module):
    def __init__(self, args, use_batch_norm=True, div_flow=20):
        r"""FlowNet2 C module. Check out the FlowNet2 paper for more details
        https://arxiv.org/abs/1612.01925

        Args:
            args (obj): Network initialization arguments
            use_batch_norm (bool): Use batch norm or not. Default is true.
            div_flow (int): Flow devision factor. Default is 20.
        """
        super(FlowNetC, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.div_flow = div_flow

        self.conv1 = conv(self.use_batch_norm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.use_batch_norm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.use_batch_norm, 128, 256, kernel_size=5,
                          stride=2)
        self.conv_redir = conv(self.use_batch_norm, 256, 32,
                               kernel_size=1, stride=1)
        self.args = args
        # if args.fp16:
        #     self.corr = nn.Sequential(
        #         tofp32(),
        #         correlation.Correlation(pad_size=20, kernel_size=1,
        #                                 max_displacement=20, stride1=1,
        #                                 stride2=2, corr_multiply=1),
        #         tofp16())
        # else:
        self.corr = correlation.Correlation(pad_size=20, kernel_size=1,
                                            max_displacement=20, stride1=1,
                                            stride2=2, corr_multiply=1)

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(self.use_batch_norm, 473, 256)
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
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)

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
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3::, :, :]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        if self.args.fp16:
            out_corr = self.corr(tofp32()(out_conv3a),
                                 tofp32()(out_conv3b))  # False
            out_corr = tofp16()(out_corr)
        else:
            out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

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
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,
