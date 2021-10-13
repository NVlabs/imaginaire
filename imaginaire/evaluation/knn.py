# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch

from imaginaire.evaluation.common import compute_nn


def _get_1nn_acc(data_x, data_y, k=1):
    device = data_x.device
    n0 = data_x.size(0)
    n1 = data_y.size(0)
    data_all = torch.cat((data_x, data_y), dim=0)
    val, idx = compute_nn(data_all, k)
    label = torch.cat((torch.ones(n0, device=device),
                       torch.zeros(n1, device=device)))

    count = torch.zeros(n0 + n1, device=device)
    for i in range(0, k):
        count = count + label.index_select(0, idx[:, i])
    pred = torch.ge(count, (float(k) / 2) *
                    torch.ones(n0 + n1, device=device)).float()

    tp = (pred * label).sum()
    fp = (pred * (1 - label)).sum()
    fn = ((1 - pred) * label).sum()
    tn = ((1 - pred) * (1 - label)).sum()
    acc_r = (tp / (tp + fn)).item()
    acc_f = (tn / (tn + fp)).item()
    acc = torch.eq(label, pred).float().mean().item()

    return {'1NN_acc': acc,
            '1NN_acc_real': acc_r,
            '1NN_acc_fake': acc_f}
