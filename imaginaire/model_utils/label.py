# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch


def make_one_hot(cfg, is_inference, data):
    r"""Convert appropriate image data types to one-hot representation.

    Args:
        data (dict): Dict containing data_type as key, with each value
            as a list of torch.Tensors.
    Returns:
        data (dict): same as input data, but with one-hot for selected
        types.
    """
    assert hasattr(cfg, 'one_hot_num_classes')
    num_classes = getattr(cfg, 'one_hot_num_classes')
    use_dont_care = getattr(cfg, 'use_dont_care', False)
    for data_type, data_type_num_classes in num_classes.items():
        if data_type in data.keys():
            data[data_type] = _encode_onehot(data[data_type] * 255.0, data_type_num_classes, use_dont_care).float()
    return data


def concat_labels(cfg, is_inference, data):
    assert hasattr(cfg, 'input_labels')
    input_labels = getattr(cfg, 'input_labels')
    dataset_type = getattr(cfg, 'type')

    # Package output.
    labels = []
    for data_type in input_labels:
        label = data.pop(data_type)
        labels.append(label)
    if not ('video' in dataset_type):
        data['label'] = torch.cat(labels, dim=0)
    else:
        data['label'] = torch.cat(labels, dim=1)
    return data


def concat_few_shot_labels(cfg, is_inference, data):
    assert hasattr(cfg, 'input_few_shot_labels')
    input_labels = getattr(cfg, 'input_few_shot_labels')
    dataset_type = getattr(cfg, 'type')

    # Package output.
    labels = []
    for data_type in input_labels:
        label = data.pop(data_type)
        labels.append(label)
    if not ('video' in dataset_type):
        data['few_shot_label'] = torch.cat(labels, dim=0)
    else:
        data['few_shot_label'] = torch.cat(labels, dim=1)
    return data


def move_dont_care(cfg, is_inference, data):
    assert hasattr(cfg, 'move_dont_care')
    move_dont_care = getattr(cfg, 'move_dont_care')
    for data_type, data_type_num_classes in move_dont_care.items():
        label_map = data[data_type] * 255.0
        label_map[label_map < 0] = data_type_num_classes
        label_map[label_map >= data_type_num_classes] = data_type_num_classes
        data[data_type] = label_map / 255.0
    return data


def _encode_onehot(label_map, num_classes, use_dont_care):
    r"""Make input one-hot.

    Args:
        label_map (torch.Tensor): (C, H, W) tensor containing indices.
        num_classes (int): Number of labels to expand tensor to.
        use_dont_care (bool): Use the dont care label or not?
    Returns:
        output (torch.Tensor): (num_classes, H, W) one-hot tensor.
    """
    # All labels lie in [0. num_classes - 1].
    # Encode dont care as num_classes.
    label_map[label_map < 0] = num_classes
    label_map[label_map >= num_classes] = num_classes

    size = label_map.size()
    output_size = (num_classes + 1, size[-2], size[-1])
    output = torch.zeros(*output_size)
    if label_map.dim() == 4:
        output = output.unsqueeze(0).repeat(label_map.size(0), 1, 1, 1)
        output = output.scatter_(1, label_map.data.long(), 1.0)
        if not use_dont_care:
            output = output[:, :num_classes, ...]
    else:
        output = output.scatter_(0, label_map.data.long(), 1.0)
        if not use_dont_care:
            output = output[:num_classes, ...]
    return output
