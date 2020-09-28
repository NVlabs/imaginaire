# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch.nn import init
import torch.nn as nn
import resample2d
import channelnorm
import numpy as np
from imaginaire.third_party.flow_net.flownet2.networks import flownet_c
from imaginaire.third_party.flow_net.flownet2.networks import flownet_s
from imaginaire.third_party.flow_net.flownet2.networks import flownet_sd
from imaginaire.third_party.flow_net.flownet2.networks import flownet_fusion
from imaginaire.third_party.flow_net.flownet2.networks.submodules import \
    tofp16, tofp32
'Parameter count = 162,518,834'


class FlowNet2(nn.Module):
    def __init__(self, args, use_batch_norm=False, div_flow=20.):
        super(FlowNet2, self).__init__()
        self.batch_norm = use_batch_norm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = channelnorm.ChannelNorm()
        # First Block (FlowNetC)
        self.flownetc = flownet_c.FlowNetC(
            args, use_batch_norm=self.batch_norm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)
        self.args = args
        # if args.fp16:
        #     self.resample1 = nn.Sequential(
        #         tofp32(), resample2d.Resample2d(), tofp16())
        # else:
        self.resample1 = resample2d.Resample2d()
        # Block (FlowNetS1)
        self.flownets_1 = flownet_s.FlowNetS(
            args, use_batch_norm=self.batch_norm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)
        # if args.fp16:
        #     self.resample2 = nn.Sequential(
        #         tofp32(), resample2d.Resample2d(), tofp16())
        # else:
        self.resample2 = resample2d.Resample2d()
        # Block (FlowNetS2)
        self.flownets_2 = flownet_s.FlowNetS(
            args, use_batch_norm=self.batch_norm)
        # Block (FlowNetSD)
        self.flownets_d = flownet_sd.FlowNetSD(
            args, use_batch_norm=self.batch_norm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        # if args.fp16:
        #     self.resample3 = nn.Sequential(
        #         tofp32(), resample2d.Resample2d(), tofp16())
        # else:
        self.resample3 = resample2d.Resample2d()
        # if args.fp16:
        #     self.resample4 = nn.Sequential(
        #         tofp32(), resample2d.Resample2d(), tofp16())
        # else:
        self.resample4 = resample2d.Resample2d()
        # Block (FLowNetFusion)
        self.flownetfusion = flownet_fusion.FlowNetFusion(
            args, use_batch_norm=self.batch_norm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        height, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([height, width])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.)
        for i in range(min_dim):
            weight.data[i, i, :, :] = torch.from_numpy(bilinear)
        return

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        # warp img1 to img0;
        # magnitude of diff between img0 and and warped_img1,
        if self.args.fp16:
            resampled_img1 = self.resample1(tofp32()(x[:, 3:, :, :]),
                                            flownetc_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            (x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0),
            dim=1)
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        # warp img1 to img0 using flownets1;
        # magnitude of diff between img0 and and warped_img1
        if self.args.fp16:
            resampled_img1 = self.resample2(tofp32()(x[:, 3:, :, :]),
                                            flownets1_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat(
            (x,
             resampled_img1,
             flownets1_flow /
             self.div_flow,
             norm_diff_img0),
            dim=1)
        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)
        if self.args.fp16:
            diff_flownets2_flow = self.resample4(tofp32()(x[:, 3:, :, :]),
                                                 flownets2_flow)
            diff_flownets2_flow = tofp16()(diff_flownets2_flow)
        else:
            diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        diff_flownets2_img1 = self.channelnorm(
            (x[:, :3, :, :] - diff_flownets2_flow))
        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        if self.args.fp16:
            diff_flownetsd_flow = self.resample3(tofp32()(x[:, 3:, :, :]),
                                                 flownetsd_flow)
            diff_flownetsd_flow = tofp16()(diff_flownetsd_flow)
        else:
            diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        diff_flownetsd_img1 = self.channelnorm(
            (x[:, :3, :, :] - diff_flownetsd_flow))
        # concat img1 flownetsd, flownets2, norm_flownetsd,
        # norm_flownets2, diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow,
                             norm_flownetsd_flow, norm_flownets2_flow,
                             diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)
        return flownetfusion_flow


class FlowNet2C(flownet_c.FlowNetC):
    def __init__(self, args, use_batch_norm=False, div_flow=20):
        super(
            FlowNet2C,
            self).__init__(
            args,
            use_batch_norm=use_batch_norm,
            div_flow=20)
        self.rgb_max = args.rgb_max

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)
        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        # Merge streams
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
            return self.upsample1(flow2 * self.div_flow)


class FlowNet2S(flownet_s.FlowNetS):
    def __init__(self, args, use_batch_norm=False, div_flow=20):
        super(FlowNet2S, self).__init__(args, input_channels=6,
                                        use_batch_norm=use_batch_norm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)
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
            return self.upsample1(flow2 * self.div_flow)


class FlowNet2SD(flownet_sd.FlowNetSD):
    def __init__(self, args, use_batch_norm=False, div_flow=20):
        super(FlowNet2SD, self).__init__(args, use_batch_norm=use_batch_norm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return self.upsample1(flow2 * self.div_flow)


class FlowNet2CS(nn.Module):
    def __init__(self, args, use_batch_norm=False, div_flow=20.):
        super(FlowNet2CS, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = channelnorm.ChannelNorm()
        # First Block (FlowNetC)
        self.flownetc = flownet_c.FlowNetC(
            args, use_batch_norm=self.use_batch_norm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)
        self.args = args
        # if args.fp16:
        #     self.resample1 = nn.Sequential(
        #         tofp32(), resample2d.Resample2d(), tofp16())
        # else:
        self.resample1 = resample2d.Resample2d()
        # Block (FlowNetS1)
        self.flownets_1 = flownet_s.FlowNetS(
            args, use_batch_norm=self.use_batch_norm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        # warp img1 to img0;
        # magnitude of diff between img0 and and warped_img1,
        if self.args.fp16:
            resampled_img1 = self.resample1(tofp32()(x[:, 3:, :, :]),
                                            flownetc_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            (x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0),
            dim=1)
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        return flownets1_flow


class FlowNet2CSS(nn.Module):
    def __init__(self, args, use_batch_norm=False, div_flow=20.):
        super(FlowNet2CSS, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = channelnorm.ChannelNorm()
        # First Block (FlowNetC)
        self.flownetc = flownet_c.FlowNetC(
            args, use_batch_norm=self.use_batch_norm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)
        self.args = args
        # if args.fp16:
        #     self.resample1 = nn.Sequential(
        #         tofp32(), resample2d.Resample2d(), tofp16())
        # else:
        self.resample1 = resample2d.Resample2d()
        # Block (FlowNetS1)
        self.flownets_1 = flownet_s.FlowNetS(
            args, use_batch_norm=self.use_batch_norm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)
        # if args.fp16:
        #     self.resample2 = nn.Sequential(
        #         tofp32(), resample2d.Resample2d(), tofp16())
        # else:
        self.resample2 = resample2d.Resample2d()
        # Block (FlowNetS2)
        self.flownets_2 = flownet_s.FlowNetS(
            args, use_batch_norm=self.use_batch_norm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest',
                                     align_corners=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        # Warp img1 to img0;
        # Magnitude of diff between img0 and and warped_img1,
        if self.args.fp16:
            resampled_img1 = self.resample1(tofp32()(x[:, 3:, :, :]),
                                            flownetc_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            (x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0),
            dim=1)
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        # Warp img1 to img0 using flownets1;
        # magnitude of diff between img0 and and warped_img1
        if self.args.fp16:
            resampled_img1 = self.resample2(tofp32()(x[:, 3:, :, :]),
                                            flownets1_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat(
            (x,
             resampled_img1,
             flownets1_flow /
             self.div_flow,
             norm_diff_img0),
            dim=1)
        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)
        return flownets2_flow
