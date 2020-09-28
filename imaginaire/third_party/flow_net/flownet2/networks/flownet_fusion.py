# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# The file is duplicated from https://github.com/NVIDIA/flownet2-pytorch
# with some modifications.
from torch.nn import init
import torch
import torch.nn as nn
from .submodules import conv, i_conv, predict_flow, deconv


class FlowNetFusion(nn.Module):
    r"""FlowNet2 Fusion module. Check out the FlowNet2 paper for more details
    https://arxiv.org/abs/1612.01925

    Args:
        args (obj): Network initialization arguments
        use_batch_norm (bool): Use batch norm or not. Default is true.
    """
    def __init__(self, args, use_batch_norm=True):
        super(FlowNetFusion, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.conv0 = conv(self.use_batch_norm, 11, 64)
        self.conv1 = conv(self.use_batch_norm, 64, 64, stride=2)
        self.conv1_1 = conv(self.use_batch_norm, 64, 128)
        self.conv2 = conv(self.use_batch_norm, 128, 128, stride=2)
        self.conv2_1 = conv(self.use_batch_norm, 128, 128)

        self.deconv1 = deconv(128, 32)
        self.deconv0 = deconv(162, 16)

        self.inter_conv1 = i_conv(self.use_batch_norm, 162, 32)
        self.inter_conv0 = i_conv(self.use_batch_norm, 82, 16)

        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)

        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

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

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensors of concatenated images.
        Returns:
            flow2 (tensor): Output flow tensors.
        """
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)

        return flow0
