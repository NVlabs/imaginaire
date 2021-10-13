# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md

# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738
# Modified from https://github.com/mit-han-lab/data-efficient-gans
import torch
import torch.nn.functional as F


def apply_diff_aug(data, keys, aug_policy, inplace=False, **kwargs):
    r"""Applies differentiable augmentation.
    Args:
        data (dict): Input data.
        keys (list of str): Keys to the data values that we want to apply
            differentiable augmentation to.
        aug_policy (str): Type of augmentation(s), ``'color'``,
            ``'translation'``, or ``'cutout'`` separated by ``','``.
    """
    if aug_policy == '':
        return data
    data_aug = data if inplace else {}
    for key, value in data.items():
        if key in keys:
            data_aug[key] = diff_aug(data[key], aug_policy, **kwargs)
        else:
            data_aug[key] = data[key]
    return data_aug


def diff_aug(x, policy='', channels_first=True, **kwargs):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x, **kwargs)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x, **kwargs):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype,
                        device=x.device) - 0.5)
    return x


def rand_saturation(x, **kwargs):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype,
                                   device=x.device) * 2) + x_mean
    return x


def rand_contrast(x, **kwargs):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype,
                                   device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125, **kwargs):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(
        x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1],
                                  device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1],
                                  device=x.device)
    # noinspection PyTypeChecker
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[
        grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5, **kwargs):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2),
                             size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2),
                             size=[x.size(0), 1, 1], device=x.device)
    # noinspection PyTypeChecker
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0,
                         max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0,
                         max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3),
                      dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_translation_scale(x, trans_r=0.125, scale_r=0.125,
                           mode='bilinear', padding_mode='reflection',
                           **kwargs):
    assert x.dim() == 4, "Input must be a 4D tensor."
    batch_size = x.size(0)

    # Identity transformation.
    theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(
        batch_size, 1, 1)

    # Translation, uniformly sampled from (-trans_r, trans_r).
    translate = \
        2 * trans_r * torch.rand(batch_size, 2, device=x.device) - trans_r
    theta[:, :, 2] += translate

    # Scaling, uniformly sampled from (1-scale_r, 1+scale_r).
    scale = \
        2 * scale_r * torch.rand(batch_size, 2, device=x.device) - scale_r
    theta[:, :, :2] += torch.diag_embed(scale)

    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(
        x.float(), grid.float(), mode=mode, padding_mode=padding_mode)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'translation_scale': [rand_translation_scale],
    'cutout': [rand_cutout],
}
