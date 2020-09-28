# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""
Modified from https://github.com/abdulfatir/gan-metrics-pytorch
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import warnings

import numpy as np
import torch

from imaginaire.evaluation.common import get_activations
from imaginaire.utils.distributed import is_master
from imaginaire.utils.distributed import master_only_print as print


def compute_kid(kid_path, data_loader, net_G,
                key_real='images', key_fake='fake_images',
                sample_size=None, preprocess=None, is_video=False,
                save_act=True, num_subsets=1, subset_size=None):
    r"""Compute the kid score.

    Args:
        kid_path (str): Location for store feature activations.
        data_loader (obj): PyTorch dataloader object.
        net_G (obj): For image generation modes, net_G is the PyTorch trainer
            network. For video generation models, net_G is the trainer
            because video generation requires more complicated processing.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        sample_size (int): How many samples to be used for computing feature
            activations.
        preprocess (func): The preprocess function to be applied to the data.
        is_video (bool): Whether we are handling video sequences.
        save_act (bool): If ``True``, saves real activations to the disk and
            reload them in the future. It might save some computation but will
            cost storage.
        num_subsets (int): Number of subsets to sample from all the samples.
        subset_size (int): Number of samples in each subset.
    Returns:
        kid (float): KID value.
    """
    print('Computing KID.')
    with torch.no_grad():
        # Get the fake activations.
        fake_act = load_or_compute_activations(None,
                                               data_loader,
                                               key_real, key_fake, net_G,
                                               sample_size, preprocess,
                                               is_video)

        # Get the ground truth activations.
        act_path = os.path.join(
            os.path.dirname(kid_path), 'activations.npy') if save_act else None
        real_act = load_or_compute_activations(act_path,
                                               data_loader,
                                               key_real, key_fake, None,
                                               sample_size, preprocess,
                                               is_video)

    if is_master():
        mmd, mmd_vars = polynomial_mmd_averages(fake_act, real_act,
                                                num_subsets,
                                                subset_size,
                                                ret_var=True)
        kid = mmd.mean()
        return kid


def compute_kid_data(kid_path, data_loader_a, data_loader_b,
                     key_a='images', key_b='images', sample_size=None,
                     is_video=False, num_subsets=1, subset_size=None):
    r"""Compute the kid score between two datasets.

    Args:
        kid_path (str): Location for store feature activations.
        data_loader_a (obj): PyTorch dataloader object for dataset a.
        data_loader_b (obj): PyTorch dataloader object for dataset b.
        key_a (str): Dictionary key value for images in the dataset a.
        key_b (str): Dictionary key value for images in the dataset b.
        sample_size (int): How many samples to be used for computing the KID.
        is_video (bool): Whether we are handling video sequences.
        num_subsets (int): Number of subsets to sample from the whole data.
        subset_size (int): Number of samples in each subset.
    Returns:
        kid (float): KID value.
    """
    if sample_size is None:
        sample_size = min(len(data_loader_a.dataset),
                          len(data_loader_b.dataset))
    print('Computing KID using {} images from both distributions.'.
          format(sample_size))
    with torch.no_grad():
        path_a = os.path.join(os.path.dirname(kid_path),
                              'activations_a.npz')
        path_b = os.path.join(os.path.dirname(kid_path),
                              'activations_b.npz')
        act_a = load_or_compute_activations(path_a, data_loader_a,
                                            key_a, key_a,
                                            sample_size=sample_size,
                                            is_video=is_video)
        act_b = load_or_compute_activations(path_b, data_loader_b,
                                            key_b, key_b,
                                            sample_size=sample_size,
                                            is_video=is_video)

        if is_master():
            mmd, mmd_vars = polynomial_mmd_averages(act_a, act_b,
                                                    num_subsets,
                                                    subset_size,
                                                    ret_var=True)
            kid = mmd.mean()
            return kid
        else:
            return None


def load_or_compute_activations(act_path, data_loader, key_real, key_fake,
                                generator=None, sample_size=None,
                                preprocess=None, is_video=False):
    r"""Load mean and covariance from saved npy file if exists. Otherwise,
    compute the mean and covariance.

    Args:
        act_path (str or None): Location for the numpy file to store or to load
            the statistics.
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int): How many samples to be used for computing the KID.
        preprocess (func): The preprocess function to be applied to the data.
        is_video (bool): Whether we are handling video sequences.
    Returns:
        mean (tensor): Mean vector.
        cov (tensor): Covariance matrix.
    """
    if is_video:
        raise NotImplementedError("Video KID is not currently supported.")
    if act_path is not None and os.path.exists(act_path):
        print('Load Inception activations from {}'.format(act_path))
        act = np.load(act_path)
    else:
        act = get_activations(data_loader, key_real, key_fake,
                              generator, sample_size, preprocess)
        if act_path is not None and is_master():
            print('Save Inception activations to {}'.format(act_path))
            np.save(act_path, act)
    return act


def polynomial_mmd_averages(codes_g, codes_r, n_subsets, subset_size,
                            ret_var=True, **kernel_args):
    r"""Computes MMD between two sets of features using polynomial kernels. It
    performs a number of repetitions of subset sampling without replacement.

    Args:
        codes_g (Tensor): Feature activations of generated images.
        codes_r (Tensor): Feature activations of real images.
        n_subsets (int): The number of subsets.
        subset_size (int): The number of samples in each subset.
        ret_var (bool): If ``True``, returns both mean and variance of MMDs,
            otherwise only returns the mean.
    Returns:
        (tuple):
          - mmds (Tensor): Mean of MMDs.
          - mmd_vars (Tensor): Variance of MMDs.
    """
    codes_g = torch.tensor(codes_g, device=torch.device('cuda'))
    codes_r = torch.tensor(codes_r, device=torch.device('cuda'))
    mmds = np.zeros(n_subsets)
    if ret_var:
        mmd_vars = np.zeros(n_subsets)
    choice = np.random.choice

    if subset_size is None:
        subset_size = min(len(codes_r), len(codes_r))
        print("Subset size not provided, "
              "setting it to the data size ({}).".format(subset_size))
    if subset_size > len(codes_g) or subset_size > len(codes_r):
        subset_size = min(len(codes_r), len(codes_r))
        warnings.warn(
            "Subset size is large than the actual data size, "
            "setting it to the data size ({}).".format(subset_size))

    for i in range(n_subsets):
        g = codes_g[choice(len(codes_g), subset_size, replace=False)]
        r = codes_r[choice(len(codes_r), subset_size, replace=False)]
        o = polynomial_mmd(g, r, **kernel_args, ret_var=ret_var)
        if ret_var:
            mmds[i], mmd_vars[i] = o
        else:
            mmds[i] = o
    return (mmds, mmd_vars) if ret_var else mmds


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1.):
    r"""Compute the polynomial kernel between X and Y"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    if Y is None:
        Y = X

    # K = safe_sparse_dot(X, Y.T, dense_output=True)
    K = torch.matmul(X, Y.t())
    K *= gamma
    K += coef0
    K = K**degree
    return K


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   ret_var=True):
    r"""Computes MMD between two sets of features using polynomial kernels. It
    performs a number of repetitions of subset sampling without replacement.

    Args:
        codes_g (Tensor): Feature activations of generated images.
        codes_r (Tensor): Feature activations of real images.
        degree (int): The degree of the polynomial kernel.
        gamma (float or None): Scale of the polynomial kernel.
        coef0 (float or None): Bias of the polynomial kernel.
        ret_var (bool): If ``True``, returns both mean and variance of MMDs,
            otherwise only returns the mean.
    Returns:
        (tuple):
          - mmds (Tensor): Mean of MMDs.
          - mmd_vars (Tensor): Variance of MMDs.
    """
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY, ret_var=ret_var)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', ret_var=True):
    r"""Based on
    https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    but changed to not compute the full kernel matrix at once
    """

    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = torch.diagonal(K_XX)
        diag_Y = torch.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)
    K_XY_sums_1 = K_XY.sum(dim=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - torch.trace(K_XY)) / (m * (m - 1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m ** 4 * K_XY_sum ** 2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m ** 4 * K_XY_sum ** 2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2.cpu().numpy(), var_est.cpu().numpy()


def _sqn(arr):
    r"""Squared norm."""
    flat = arr.view(-1)
    return flat.dot(flat)
