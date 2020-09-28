# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from imaginaire.third_party.flow_net.flownet2 import models as \
    flownet2_models
from imaginaire.third_party.flow_net.flownet2.utils import tools \
    as flownet2_tools
from imaginaire.model_utils.fs_vid2vid import resample
from imaginaire.utils.io import get_checkpoint


class FlowNet(nn.Module):
    def __init__(self, pretrained=True, fp16=False):
        super().__init__()
        flownet2_args = types.SimpleNamespace()
        setattr(flownet2_args, 'fp16', fp16)
        setattr(flownet2_args, 'rgb_max', 1.0)
        if fp16:
            print('FlowNet2 is running in fp16 mode.')
        self.flowNet = flownet2_tools.module_to_dict(flownet2_models)[
            'FlowNet2'](flownet2_args).to('cuda')
        if pretrained:
            flownet2_path = get_checkpoint('flownet2.pth.tar',
                                           '1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da')
            checkpoint = torch.load(flownet2_path,
                                    map_location=torch.device('cpu'))
            self.flowNet.load_state_dict(checkpoint['state_dict'])
        self.flowNet.eval()

    def forward(self, input_A, input_B):
        size = input_A.size()
        assert(len(size) == 4 or len(size) == 5 or len(size) == 6)
        if len(size) >= 5:
            if len(size) == 5:
                b, n, c, h, w = size
            else:
                b, t, n, c, h, w = size
            input_A = input_A.contiguous().view(-1, c, h, w)
            input_B = input_B.contiguous().view(-1, c, h, w)
            flow, conf = self.compute_flow_and_conf(input_A, input_B)
            if len(size) == 5:
                return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
            else:
                return flow.view(b, t, n, 2, h, w), conf.view(b, t, n, 1, h, w)
        else:
            return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        assert(im1.size()[1] == 3)
        assert(im1.size() == im2.size())
        old_h, old_w = im1.size()[2], im1.size()[3]
        new_h, new_w = old_h // 64 * 64, old_w // 64 * 64
        if old_h != new_h:
            im1 = F.interpolate(im1, size=(new_h, new_w), mode='bilinear',
                                align_corners=False)
            im2 = F.interpolate(im2, size=(new_h, new_w), mode='bilinear',
                                align_corners=False)
        data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)
        with torch.no_grad():
            flow1 = self.flowNet(data1)
        # img_diff = torch.sum(abs(im1 - resample(im2, flow1)),
        #                      dim=1, keepdim=True)
        # conf = torch.clamp(1 - img_diff, 0, 1)

        conf = (self.norm(im1 - resample(im2, flow1)) < 0.02).float()

        # data2 = torch.cat([im2.unsqueeze(2), im1.unsqueeze(2)], dim=2)
        # with torch.no_grad():
        #     flow2 = self.flowNet(data2)
        # warped_flow2 = resample(flow2, flow1)
        # flow_sum = self.norm(flow1 + warped_flow2)
        # disocc = flow_sum > (0.05 * (self.norm(flow1) +
        # self.norm(warped_flow2)) + 0.5)
        # conf = 1 - disocc.float()

        if old_h != new_h:
            flow1 = F.interpolate(flow1, size=(old_h, old_w), mode='bilinear',
                                  align_corners=False) * old_h / new_h
            conf = F.interpolate(conf, size=(old_h, old_w), mode='bilinear',
                                 align_corners=False)
        return flow1, conf

    def norm(self, t):
        return torch.sum(t * t, dim=1, keepdim=True)
