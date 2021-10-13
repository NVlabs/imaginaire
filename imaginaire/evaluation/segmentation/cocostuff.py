# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from torch import nn
from torch.nn import functional as F
import torch.hub


class DeepLabV2(nn.Module):
    def __init__(self, n_classes=182, image_size=512, use_dont_care=True):
        super(DeepLabV2, self).__init__()
        self.model = torch.hub.load(
            "kazuto1011/deeplab-pytorch", "deeplabv2_resnet101",
            pretrained=False, n_classes=182
        )
        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/kazuto1011/deeplab-pytorch/releases/download/'
            'v1.0/deeplabv2_resnet101_msc-cocostuff164k-100000.pth',
            map_location="cpu"
        )
        self.model.load_state_dict(state_dict)

        self.image_size = image_size
        # self.mean = torch.tensor([122.675, 116.669, 104.008], device="cuda")
        self.mean = torch.tensor([104.008, 116.669, 122.675], device="cuda")
        self.n_classes = n_classes
        self.use_dont_care = use_dont_care

    def forward(self, images, align_corners=True):
        scale = self.image_size / max(images.shape[2:])
        images = F.interpolate(
            images, scale_factor=scale, mode='bilinear',
            align_corners=align_corners
        )
        images = 255 * 0.5 * (images + 1)  # (-1, 1) -> (0, 255)
        images = images.flip(1)  # RGB to BGR
        images -= self.mean[None, :, None, None]
        _, _, H, W = images.shape

        logits = self.model(images)
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear",
            align_corners=align_corners
        )
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        return pred
