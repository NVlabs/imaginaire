# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import boto3
import torch
from torch import nn, distributed as dist
from torch.nn import functional as F
from torch.distributed import barrier

from imaginaire.utils.distributed import is_local_master
from .clip import build_model
from ...utils.io import download_file_from_google_drive


def get_image_encoder(aws_credentials=None):
    if dist.is_initialized() and not is_local_master():
        # Make sure only the first process in distributed training downloads the model, and the others use the cache.
        barrier()

    # Load the CLIP image encoder.
    print("Loading CLIP image encoder.")
    model_path = os.path.join(torch.hub.get_dir(), 'checkpoints', 'ViT-B-32.pt')
    if not os.path.exists(model_path):
        if aws_credentials is not None:
            s3 = boto3.client('s3', **aws_credentials)
            s3.download_file('lpi-poe', 'model_zoo/ViT-B-32.pt', model_path)
        else:
            download_file_from_google_drive("1Ri5APYM34A_IjG4F3Admutsf2oUwDjfW", model_path)
    model = torch.load(model_path, map_location='cpu')

    if dist.is_initialized() and is_local_master():
        # Make sure only the first process in distributed training downloads the model, and the others use the cache.
        barrier()

    encoder = build_model(model).cuda()
    return ImageEncoder(encoder)


class ImageEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.model = encoder
        self.image_size = self.model.visual.input_resolution
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda")
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda")

    @torch.no_grad()
    def forward(self, data, fake_images, align_corners=True):
        images = 0.5 * (1 + fake_images)
        images = F.interpolate(images, (self.image_size, self.image_size), mode='bicubic', align_corners=align_corners)
        images.clamp_(0, 1)
        images = (images - self.mean[None, :, None, None]) / (self.std[None, :, None, None])
        image_code = self.model.encode_image(images)
        return torch.cat((image_code, data['captions-clip']), dim=1)
