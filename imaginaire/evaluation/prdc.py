# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""
Modified from https://github.com/clovaai/generative-evaluation-prdc
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os

import torch

from imaginaire.utils.distributed import is_master
from imaginaire.utils.distributed import master_only_print as print

from .common import load_or_compute_activations, compute_pairwise_distance, \
    compute_nn


@torch.no_grad()
def compute_prdc(prdc_path, data_loader, net_G,
                 key_real='images', key_fake='fake_images',
                 real_act=None, fake_act=None,
                 sample_size=None, save_act=True, k=10, **kwargs):
    r"""Compute precision diversity curve

    Args:

    """
    print('Computing PRDC.')
    act_path = os.path.join(
        os.path.dirname(prdc_path), 'activations_real.npy'
    ) if save_act else None

    # Get the fake activations.
    if fake_act is None:
        fake_act = load_or_compute_activations(None,
                                               data_loader,
                                               key_real, key_fake, net_G,
                                               sample_size=sample_size,
                                               **kwargs)
    else:
        print(f"Using precomputed activations of size {fake_act.shape}.")

    # Get the ground truth activations.
    if real_act is None:
        real_act = load_or_compute_activations(act_path,
                                               data_loader,
                                               key_real, key_fake, None,
                                               sample_size=sample_size,
                                               **kwargs)
    else:
        print(f"Using precomputed activations of size {real_act.shape}.")

    if is_master():
        prdc_data = _get_prdc(real_act, fake_act, k)
        return \
            prdc_data['precision'], prdc_data['recall'], \
            prdc_data['density'], prdc_data['coverage']
    else:
        return None, None, None, None


def get_kth_value(unsorted, k, dim=-1):
    r"""

    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = torch.topk(unsorted, k, dim=dim, largest=False)[1]
    k_smallests = torch.gather(unsorted, dim=dim, index=indices)
    kth_values = k_smallests.max(dim=dim)[0]
    return kth_values


def _get_prdc(real_features, fake_features, nearest_k):
    r"""
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    real_nearest_neighbour_distances, _ = compute_nn(
        real_features, nearest_k)
    real_nearest_neighbour_distances = \
        real_nearest_neighbour_distances.max(dim=-1)[0].cpu()
    fake_nearest_neighbour_distances, _ = compute_nn(
        fake_features, nearest_k)
    fake_nearest_neighbour_distances = \
        fake_nearest_neighbour_distances.max(dim=-1)[0].cpu()
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            torch.unsqueeze(real_nearest_neighbour_distances, dim=1)
    ).any(dim=0).float().mean().item()

    recall = (
            distance_real_fake <
            torch.unsqueeze(fake_nearest_neighbour_distances, dim=0)
    ).any(dim=1).float().mean().item()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            torch.unsqueeze(real_nearest_neighbour_distances, dim=1)
    ).sum(dim=0).float().mean().item()

    # noinspection PyUnresolvedReferences
    coverage = (
            distance_real_fake.min(dim=1)[0] <
            real_nearest_neighbour_distances
    ).float().mean().item()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)
