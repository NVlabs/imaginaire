# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from imaginaire.utils.distributed import get_world_size, get_rank, \
    dist_all_reduce_tensor


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        all_grads = torch.stack(grads)
        all_grads = dist_all_reduce_tensor(all_grads, reduce='sum')
        grad_out[:] = all_grads[get_rank()]
        return grad_out


class InfoNCELoss(nn.Module):
    def __init__(self,
                 temperature=0.07,
                 gather_distributed=True,
                 learn_temperature=True,
                 single_direction=False,
                 flatten=True):
        super(InfoNCELoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.tensor([math.log(1/temperature)]))
        self.logit_scale.requires_grad = learn_temperature
        self.gather_distributed = gather_distributed
        self.single_direction = single_direction
        self.flatten = flatten

    def forward(self, features_a, features_b, gather_distributed=None, eps=1e-8):
        if gather_distributed is None:
            gather_distributed = self.gather_distributed

        if features_a is None or features_b is None:
            return torch.tensor(0, device='cuda'), torch.tensor(0, device='cuda')

        bs_a, bs_b = features_a.size(0), features_b.size(0)
        if self.flatten:
            features_a, features_b = features_a.reshape(bs_a, -1), features_b.reshape(bs_b, -1)
        else:
            features_a = features_a.reshape(bs_a, features_a.size(1), -1).mean(-1)
            features_b = features_b.reshape(bs_b, features_b.size(1), -1).mean(-1)

        # Temperature clipping.
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)

        # normalized features
        features_a = features_a / (features_a.norm(dim=1, keepdim=True) + eps)
        features_b = features_b / (features_b.norm(dim=1, keepdim=True) + eps)

        loss_a = self._forward_single_direction(features_a, features_b, gather_distributed)
        if self.single_direction:
            return loss_a
        else:
            loss_b = self._forward_single_direction(features_b, features_a, gather_distributed)
            return loss_a + loss_b

    def _forward_single_direction(
            self, features_a, features_b, gather_distributed):
        bs_a = features_a.shape[0]
        logit_scale = self.logit_scale.exp()
        if get_world_size() > 1 and gather_distributed:
            gather_features_b = torch.cat(GatherLayer.apply(features_b))
            gather_labels_a = torch.arange(bs_a, device='cuda') + get_rank() * bs_a
            logits_a = logit_scale * features_a @ gather_features_b.t()
        else:
            gather_labels_a = torch.arange(bs_a, device='cuda')
            logits_a = logit_scale * features_a @ features_b.t()
        loss_a = F.cross_entropy(logits_a, gather_labels_a)
        return loss_a
