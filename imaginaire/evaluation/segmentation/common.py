# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import boto3
import torch
from torch import nn, distributed as dist
from torch.nn import functional as F

from imaginaire.utils.distributed import is_local_master
from imaginaire.utils.io import download_file_from_google_drive


def get_segmentation_hist_model(dataset_name, aws_credentials=None):
    if dist.is_initialized() and not is_local_master():
        # Make sure only the first process in distributed training downloads
        # the model, and the others will use the cache
        # noinspection PyUnresolvedReferences
        torch.distributed.barrier()

    # Load the segmentation network.
    if dataset_name == "celebamask_hq":
        from imaginaire.evaluation.segmentation.celebamask_hq import Unet
        seg_network = Unet()
        os.makedirs(os.path.join(torch.hub.get_dir(), 'checkpoints'), exist_ok=True)
        model_path = os.path.join(os.path.join(torch.hub.get_dir(), 'checkpoints'), "celebamask_hq.pt")
        if not os.path.exists(model_path):
            if aws_credentials is not None:
                s3 = boto3.client('s3', **aws_credentials)
                s3.download_file('lpi-poe', 'model_zoo/celebamask_hq.pt', model_path)
            else:
                download_file_from_google_drive("1o1m-eT38zNCIFldcRaoWcLvvBtY8S4W3", model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        seg_network.load_state_dict(state_dict)
    elif dataset_name == "cocostuff" or dataset_name == "getty":
        from imaginaire.evaluation.segmentation.cocostuff import DeepLabV2
        seg_network = DeepLabV2()
    else:
        print(f"No segmentation network for {dataset_name} was found.")
        return None

    if dist.is_initialized() and is_local_master():
        # Make sure only the first process in distributed training downloads
        # the model, and the others will use the cache
        # noinspection PyUnresolvedReferences
        torch.distributed.barrier()

    if seg_network is not None:
        seg_network = seg_network.to('cuda').eval()

    return SegmentationHistModel(seg_network)


class SegmentationHistModel(nn.Module):
    def __init__(self, seg_network):
        super().__init__()
        self.seg_network = seg_network

    def forward(self, data, fake_images, align_corners=True):
        pred = self.seg_network(fake_images, align_corners=align_corners)
        gt = data["segmaps"]
        gt = gt * 255.0
        gt = gt.long()
        # print(fake_images.shape, fake_images.min(), fake_images.max())
        # print(gt.shape, gt.min(), gt.max())
        # exit()
        return compute_hist(pred, gt, self.seg_network.n_classes, self.seg_network.use_dont_care)


def compute_hist(pred, gt, n_classes, use_dont_care):
    _, H, W = pred.size()
    gt = F.interpolate(gt.float(), (H, W), mode="nearest").long().squeeze(1)
    ignore_idx = n_classes if use_dont_care else -1
    all_hist = []
    for cur_pred, cur_gt in zip(pred, gt):
        keep = torch.logical_not(cur_gt == ignore_idx)
        merge = cur_pred[keep] * n_classes + cur_gt[keep]
        hist = torch.bincount(merge, minlength=n_classes ** 2)
        hist = hist.view((n_classes, n_classes))
        all_hist.append(hist)
    all_hist = torch.stack(all_hist)
    return all_hist


def get_miou(hist, eps=1e-8):
    hist = hist.sum(0)
    IOUs = torch.diag(hist) / (
            torch.sum(hist, dim=0, keepdim=False) + torch.sum(hist, dim=1, keepdim=False) - torch.diag(hist) + eps)
    mIOU = 100 * torch.mean(IOUs).item()
    return {"seg_mIOU": mIOU}
