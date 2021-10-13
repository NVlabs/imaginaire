# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from collections import namedtuple

import torch
from torch import nn, distributed as dist
import torchvision.models as tv
from torch.distributed import barrier

from imaginaire.utils.distributed import is_local_master


def get_lpips_model():
    if dist.is_initialized() and not is_local_master():
        # Make sure only the first process in distributed training downloads the model, and the others use the cache.
        barrier()

    model = LPIPSNet().cuda()

    if dist.is_initialized() and is_local_master():
        # Make sure only the first process in distributed training downloads the model, and the others use the cache.
        barrier()
    return model


# Learned perceptual network, modified from https://github.com/richzhang/PerceptualSimilarity

def normalize_tensor(in_feat, eps=1e-5):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)


class NetLinLayer(nn.Module):
    """ A single linear layer used as placeholder for LPIPS learnt weights """

    def __init__(self, dim):
        super(NetLinLayer, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, inp):
        out = self.weight * inp
        return out


class ScalingLayer(nn.Module):
    # For rescaling the input to vgg16
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class LPIPSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LPNet()

    @torch.no_grad()
    def forward(self, fake_images, fake_images_another, align_corners=True):
        features, shape = self._forward_single(fake_images)
        features_another, _ = self._forward_single(fake_images_another)
        result = 0
        for i, g_feat in enumerate(features):
            cur_diff = torch.sum((g_feat - features_another[i]) ** 2, dim=1) / (shape[i] ** 2)
            result += cur_diff
        return result

    def _forward_single(self, images):
        return self.model(torch.clamp(images, 0, 1))


class LPNet(nn.Module):
    def __init__(self):
        super(LPNet, self).__init__()

        self.scaling_layer = ScalingLayer()
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.L = 5
        dims = [64, 128, 256, 512, 512]
        self.lins = nn.ModuleList([NetLinLayer(dims[i]) for i in range(self.L)])

        weights = torch.hub.load_state_dict_from_url(
            'https://github.com/niopeng/CAM-Net/raw/main/code/models/weights/v0.1/vgg.pth'
        )
        for i in range(self.L):
            self.lins[i].weight.data = torch.sqrt(weights["lin%d.model.1.weight" % i])

    def forward(self, in0, avg=False):
        in0 = 2 * in0 - 1
        in0_input = self.scaling_layer(in0)
        outs0 = self.net.forward(in0_input)
        feats0 = {}
        shapes = []
        res = []

        for kk in range(self.L):
            feats0[kk] = normalize_tensor(outs0[kk])

        if avg:
            res = [self.lins[kk](feats0[kk]).mean([2, 3], keepdim=False) for kk in range(self.L)]
        else:
            for kk in range(self.L):
                cur_res = self.lins[kk](feats0[kk])
                shapes.append(cur_res.shape[-1])
                res.append(cur_res.reshape(cur_res.shape[0], -1))

        return res, shapes


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out
