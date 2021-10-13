# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os
import numpy as np
import torch
from scipy import linalg

from imaginaire.evaluation.common import load_or_compute_activations
from imaginaire.utils.distributed import is_master
from imaginaire.utils.distributed import master_only_print as print


@torch.no_grad()
def compute_fid(fid_path, data_loader, net_G,
                key_real='images', key_fake='fake_images',
                sample_size=None, preprocess=None, return_act=False,
                is_video=False, few_shot_video=False, **kwargs):
    r"""Compute the fid score.

    Args:
        fid_path (str): Location for the numpy file to store or to load the
            statistics.
        data_loader (obj): PyTorch dataloader object.
        net_G (obj): For image generation modes, net_G is the generator network.
            For video generation models, net_G is the trainer.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        sample_size (int or tuple): How many samples to be used.
        preprocess (func): The preprocess function to be applied to the data.
        return_act (bool): If ``True``, also returns feature activations of
            real and fake data.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
    Returns:
        (float): FID value.
    """
    print('Computing FID.')
    act_path = os.path.join(os.path.dirname(fid_path),
                            'activations_real.npy')
    # Get the fake mean and covariance.
    fake_act = load_or_compute_activations(
        None, data_loader, key_real, key_fake, net_G,
        sample_size, preprocess, is_video=is_video,
        few_shot_video=few_shot_video, **kwargs
    )

    # Get the ground truth mean and covariance.
    real_act = load_or_compute_activations(
        act_path, data_loader, key_real, key_fake, None,
        sample_size, preprocess, is_video=is_video,
        few_shot_video=few_shot_video, **kwargs
    )

    if is_master():
        fid = _calculate_frechet_distance(
            fake_act, real_act)["FID"]
        if return_act:
            return fid, real_act, fake_act
        else:
            return fid
    elif return_act:
        return None, None, None
    else:
        return None


@torch.no_grad()
def compute_fid_data(fid_path, data_loader_a, data_loader_b,
                     key_a='images', key_b='images', sample_size=None,
                     is_video=False, few_shot_video=False, **kwargs):
    r"""Compute the fid score between two datasets.

    Args:
        fid_path (str): Location for the numpy file to store or to load the
            statistics.
        data_loader_a (obj): PyTorch dataloader object for dataset a.
        data_loader_b (obj): PyTorch dataloader object for dataset b.
        key_a (str): Dictionary key value for images in the dataset a.
        key_b (str): Dictionary key value for images in the dataset b.
        sample_size (int): How many samples to be used for computing the FID.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
    Returns:
        (float): FID value.
    """
    print('Computing FID.')
    path_a = os.path.join(os.path.dirname(fid_path),
                          'activations_a.npy')
    min_data_size = min(len(data_loader_a.dataset),
                        len(data_loader_b.dataset))
    if sample_size is None:
        sample_size = min_data_size
    else:
        sample_size = min(sample_size, min_data_size)

    act_a = load_or_compute_activations(
        path_a, data_loader_a, key_a, key_b, None,
        sample_size=sample_size, is_video=is_video,
        few_shot_video=few_shot_video, **kwargs
    )
    act_b = load_or_compute_activations(
        None, data_loader_b, key_a, key_b, None,
        sample_size=sample_size, is_video=is_video,
        few_shot_video=few_shot_video, **kwargs
    )

    if is_master():
        return _calculate_frechet_distance(act_a, act_b)["FID"]


def _calculate_frechet_distance(act_1, act_2, eps=1e-6):
    mu1 = np.mean(act_1.cpu().numpy(), axis=0)
    sigma1 = np.cov(act_1.cpu().numpy(), rowvar=False)
    mu2 = np.mean(act_2.cpu().numpy(), axis=0)
    sigma2 = np.cov(act_2.cpu().numpy(), rowvar=False)
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print('Imaginary component {}'.format(m))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return {"FID": (diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean)}
