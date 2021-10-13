# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md


def rename_inputs(cfg, is_inference, data):
    assert hasattr(cfg, 'rename_inputs')
    attr = getattr(cfg, 'rename_inputs')
    for key in attr.keys():
        value = attr[key]
        data[key] = data[value]
        # Delete the old key.
        del data[value]
    return data
