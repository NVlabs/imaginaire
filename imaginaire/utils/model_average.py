# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import copy

import torch
from torch import nn
from imaginaire.layers.weight_norm import remove_weight_norms
from imaginaire.utils.misc import requires_grad


def reset_batch_norm(m):
    r"""Reset batch norm statistics

    Args:
        m: Pytorch module
    """
    if hasattr(m, 'reset_running_stats'):
        m.reset_running_stats()


def calibrate_batch_norm_momentum(m):
    r"""Calibrate batch norm momentum

    Args:
        m: Pytorch module
    """
    if hasattr(m, 'reset_running_stats'):
        # if m._get_name() == 'SyncBatchNorm':
        if 'BatchNorm' in m._get_name():
            m.momentum = 1.0 / float(m.num_batches_tracked + 1)


class ModelAverage(nn.Module):
    r"""In this model average implementation, the spectral layers are
    absorbed in the model parameter by default. If such options are
    turned on, be careful with how you do the training. Remember to
    re-estimate the batch norm parameters before using the model.

    Args:
        module (torch nn module): Torch network.
        beta (float): Moving average weights. How much we weight the past.
        start_iteration (int): From which iteration, we start the update.
        remove_sn (bool): Whether we remove the spectral norm when we it.
    """
    def __init__(
            self, module, beta=0.9999, start_iteration=1000,
            remove_wn_wrapper=True
    ):
        super(ModelAverage, self).__init__()
        self.module = module
        # A shallow copy creates a new object which stores the reference of
        # the original elements.
        # A deep copy creates a new object and recursively adds the copies of
        # nested objects present in the original elements.
        self.averaged_model = copy.deepcopy(self.module).to('cuda')
        self.beta = beta
        self.remove_wn_wrapper = remove_wn_wrapper
        self.start_iteration = start_iteration
        # This buffer is to track how many iterations has the model been
        # trained for. We will ignore the first $(start_iterations) and start
        # the averaging after.
        self.register_buffer('num_updates_tracked',
                             torch.tensor(0, dtype=torch.long))
        self.num_updates_tracked = self.num_updates_tracked.to('cuda')
        # if self.remove_sn:
        #     # If we want to remove the spectral norm, we first copy the
        #     # weights to the moving average model.
        #     self.copy_s2t()
        #
        #     def fn_remove_sn(m):
        #         r"""Remove spectral norm."""
        #         if hasattr(m, 'weight_orig'):
        #             remove_spectral_norm(m)
        #
        #     self.averaged_model.apply(fn_remove_sn)
        #     self.dim = 0
        if self.remove_wn_wrapper:
            self.copy_s2t()

            self.averaged_model.apply(remove_weight_norms)
            self.dim = 0
        else:
            self.averaged_model.eval()

        # Averaged model does not require grad.
        requires_grad(self.averaged_model, False)

    def forward(self, *inputs, **kwargs):
        r"""PyTorch module forward function overload."""
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def update_average(self):
        r"""Update the moving average."""
        self.num_updates_tracked += 1
        if self.num_updates_tracked <= self.start_iteration:
            beta = 0.
        else:
            beta = self.beta
        source_dict = self.module.state_dict()
        target_dict = self.averaged_model.state_dict()
        for key in target_dict:
            if 'num_batches_tracked' in key:
                continue
            if self.remove_wn_wrapper:
                if key.endswith('weight'):
                    # This is a weight parameter.
                    if key + '_ori' in source_dict:
                        # This parameter has scaled lr.
                        source_param = \
                            source_dict[key + '_ori'] * \
                            source_dict[key + '_scale']
                    elif key + '_orig' in source_dict:
                        # This parameter has spectral norm
                        # but not scaled lr.
                        source_param = source_dict[key + '_orig']
                    elif key in source_dict:
                        # This parameter does not have
                        # weight normalization wrappers.
                        source_param = source_dict[key]
                    else:
                        raise ValueError(
                            f"{key} required in the averaged model but not "
                            f"found in the regular model."
                        )
                    source_param = source_param.detach()

                    if key + '_orig' in source_dict:
                        # This parameter has spectral norm.
                        source_param = self.sn_compute_weight(
                            source_param,
                            source_dict[key + '_u'],
                            source_dict[key + '_v'],
                        )
                elif key.endswith('bias') and key + '_ori' in source_dict:
                    # This is a bias parameter and has scaled lr.
                    source_param = source_dict[key + '_ori'] * \
                                   source_dict[key + '_scale']
                else:
                    # This is a normal parameter.
                    source_param = source_dict[key]
                target_dict[key].data.mul_(beta).add_(
                    source_param.data, alpha=1 - beta
                )
            else:
                target_dict[key].data.mul_(beta).add_(
                    source_dict[key].data, alpha=1 - beta
                )

    @torch.no_grad()
    def copy_t2s(self):
        r"""Copy the original weights to the moving average weights."""
        target_dict = self.module.state_dict()
        source_dict = self.averaged_model.state_dict()
        beta = 0.
        for key in source_dict:
            target_dict[key].data.copy_(
                target_dict[key].data * beta +
                source_dict[key].data * (1 - beta))

    @torch.no_grad()
    def copy_s2t(self):
        r""" Copy state_dictionary from source to target.
        Here source is the regular module and the target is the moving
        average module. Basically, we will copy weights in the regular module
        to the moving average module.
        """
        source_dict = self.module.state_dict()
        target_dict = self.averaged_model.state_dict()
        beta = 0.
        for key in source_dict:
            target_dict[key].data.copy_(
                target_dict[key].data * beta +
                source_dict[key].data * (1 - beta))

    def __repr__(self):
        r"""Returns a string that holds a printable representation of an
        object"""
        return self.module.__repr__()

    def sn_reshape_weight_to_matrix(self, weight):
        r"""Reshape weight to obtain the matrix form.

        Args:
            weight (Parameters): pytorch layer parameter tensor.

        Returns:
            (Parameters): Reshaped weight matrix
        """
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim,
                *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def sn_compute_weight(self, weight, u, v):
        r"""Compute the spectral norm normalized matrix.

        Args:
            weight (Parameters): pytorch layer parameter tensor.
            u (tensor): left singular vectors.
            v (tensor) right singular vectors

        Returns:
            (Parameters): weight parameter object.
        """
        weight_mat = self.sn_reshape_weight_to_matrix(weight)
        sigma = torch.sum(u * torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight
