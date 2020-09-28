# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import numpy as np
import torch
from scipy import linalg

from imaginaire.evaluation.common import get_activations, get_video_activations
from imaginaire.utils.distributed import is_master
from imaginaire.utils.distributed import master_only_print as print


def compute_fid(fid_path, data_loader, net_G,
                key_real='images', key_fake='fake_images',
                sample_size=None, preprocess=None,
                is_video=False, few_shot_video=False):
    r"""Compute the fid score.

    Args:
        fid_path (str): Location for the numpy file to store or to load the
            statistics.
        data_loader (obj): PyTorch dataloader object.
        net_G (obj): For image generation modes, net_G is the PyTorch trainer
            network. For video generation models, net_G is the trainer
            because video generation requires more complicated processing.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        sample_size (int or tuple): How many samples to be used.
        preprocess (func): The preprocess function to be applied to the data.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
    Returns:
        (float): FID value.
    """
    print('Computing FID.')
    with torch.no_grad():
        # Get the fake mean and covariance.
        fake_mean, fake_cov = load_or_compute_stats(fid_path,
                                                    data_loader,
                                                    key_real, key_fake, net_G,
                                                    sample_size, preprocess,
                                                    is_video, few_shot_video)
        # Get the ground truth mean and covariance.
        mean_cov_path = os.path.join(os.path.dirname(fid_path),
                                     'real_mean_cov.npz')
        real_mean, real_cov = load_or_compute_stats(mean_cov_path,
                                                    data_loader,
                                                    key_real, key_fake, None,
                                                    sample_size, preprocess,
                                                    is_video, few_shot_video)

    if is_master():
        fid = calculate_frechet_distance(
            real_mean, real_cov, fake_mean, fake_cov)
        return fid


def compute_fid_data(fid_path, data_loader_a, data_loader_b,
                     key_a='images', key_b='images', sample_size=None,
                     is_video=False, few_shot_video=False):
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
    if sample_size is None:
        sample_size = min(len(data_loader_a.dataset),
                          len(data_loader_b.dataset))
    print('Computing FID using {} images from both distributions.'.
          format(sample_size))
    with torch.no_grad():
        path_a = os.path.join(os.path.dirname(fid_path),
                              'mean_cov_a.npz')
        path_b = os.path.join(os.path.dirname(fid_path),
                              'mean_cov_b.npz')
        mean_a, cov_a = load_or_compute_stats(path_a, data_loader_a,
                                              key_a, key_a,
                                              sample_size=sample_size,
                                              is_video=is_video)
        mean_b, cov_b = load_or_compute_stats(path_b, data_loader_b,
                                              key_b, key_b,
                                              sample_size=sample_size,
                                              is_video=is_video)
    if is_master():
        fid = calculate_frechet_distance(mean_b, cov_b, mean_a, cov_a)
        return fid


def load_or_compute_stats(fid_path, data_loader, key_real, key_fake,
                          generator=None, sample_size=None, preprocess=None,
                          is_video=False, few_shot_video=False):
    r"""Load mean and covariance from saved npy file if exists. Otherwise,
    compute the mean and covariance.

    Args:
        fid_path (str): Location for the numpy file to store or to load the
            statistics.
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int or tuple): How many samples to be used.
        preprocess (func): The preprocess function to be applied to the data.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
    Returns:
        (dict):
          - mean (tensor): Mean vector.
          - cov (tensor): Covariance matrix.
    """
    if os.path.exists(fid_path):
        print('Load FID mean and cov from {}'.format(fid_path))
        npz_file = np.load(fid_path)
        mean = npz_file['mean']
        cov = npz_file['cov']
    else:
        print('Get FID mean and cov and save to {}'.format(fid_path))
        mean, cov = get_inception_mean_cov(data_loader, key_real, key_fake,
                                           generator, sample_size, preprocess,
                                           is_video, few_shot_video)
        os.makedirs(os.path.dirname(fid_path), exist_ok=True)
        if is_master():
            np.savez(fid_path, mean=mean, cov=cov)
    return mean, cov


def get_inception_mean_cov(data_loader, key_real, key_fake, generator,
                           sample_size, preprocess,
                           is_video=False, few_shot_video=False):
    r"""Load mean and covariance from saved npy file if exists. Otherwise,
    compute the mean and covariance.

    Args:
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int or tuple): How many samples to be used.
        preprocess (func): The preprocess function to be applied to the data.
        is_video (bool): Whether we are handling video sequences.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
    Returns:
        (dict):
          - mean (tensor): Mean vector.
          - cov (tensor): Covariance matrix.
    trainer (obj): PyTorch trainer network.
    """
    print('Extract mean and covariance.')
    if is_video:
        y = get_video_activations(data_loader, key_real, key_fake, generator,
                                  sample_size, preprocess, few_shot_video)
    else:
        y = get_activations(data_loader, key_real, key_fake, generator,
                            sample_size, preprocess)
    if is_master():
        m = np.mean(y, axis=0)
        s = np.cov(y, rowvar=False)
    else:
        m = None
        s = None
    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    r"""Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1: Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
        mu2: The sample mean over activations, pre-calculated on an
            representative data set.
        sigma1: The covariance matrix over activations for generated samples.
        sigma2: The covariance matrix over activations, pre-calculated on an
            representative data set.
        eps: a value added to the diagonal of cov for numerical stability.
    Returns:
        The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
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
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean)
