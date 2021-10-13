# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch


class Dataset(torch.utils.data.Dataset):
    r"""Dummy dataset, returns nothing."""

    def __init__(self, cfg, is_inference=False, is_test=False):
        super(Dataset, self).__init__()

    def __getitem__(self, index):
        return {}

    def __len__(self):
        return 65535
