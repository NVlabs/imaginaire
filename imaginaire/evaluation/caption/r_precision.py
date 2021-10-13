# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# flake8: noqa

import torch
import torch.nn.functional as F


def get_r_precision(image_text_code, eps=1e-5):
    all_image_code, all_text_code = torch.chunk(image_text_code, 2, dim=1)
    P_rates = []
    num_samples = len(all_image_code)
    assert num_samples >= 100
    for i in range(0, num_samples, 100):
        if i + 100 <= num_samples:
            cur_image_code = all_image_code[i:i + 100]
            cur_text_code = all_text_code[i:i + 100]
            cur_image_code = F.normalize(cur_image_code, dim=1, eps=eps)
            cur_text_code = F.normalize(cur_text_code, dim=1, eps=eps)
            cosine_similarities = cur_image_code @ cur_text_code.T
            top1_indices = torch.topk(cosine_similarities, dim=1, k=1)[1][:, 0]
            P_rate = torch.sum(top1_indices == torch.arange(100, device=top1_indices.device)).item()
            P_rates.append(P_rate)
    A_precision = sum(P_rates) * 1.0 / len(P_rates)
    return {"caption_rprec": A_precision}
