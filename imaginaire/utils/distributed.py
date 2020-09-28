# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools

import torch
import torch.distributed as dist


def init_dist(local_rank, backend='nccl', **kwargs):
    r"""Initialize distributed training"""
    if dist.is_available():
        if dist.is_initialized():
            return torch.cuda.current_device()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method='env://', **kwargs)


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def master_only(func):
    r"""Apply this function only to the master GPU."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r"""Simple function wrapper for the master function"""
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


@master_only
def master_only_print(*args):
    r"""master-only print"""
    print(*args)


def dist_reduce_tensor(tensor):
    r""" Reduce to rank 0 """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if get_rank() == 0:
            tensor /= world_size
    return tensor


def dist_all_reduce_tensor(tensor):
    r""" Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


def dist_all_gather_tensor(tensor):
    r""" gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    tensor_list = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)
    return tensor_list
