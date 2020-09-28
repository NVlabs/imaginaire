# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import importlib
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop, lr_scheduler

import apex
from apex import amp
from apex.optimizers import FusedAdam, FusedSGD
from imaginaire.optimizers import Fromage, Madam
from imaginaire.utils.distributed import get_rank
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.init_weight import weights_init
from imaginaire.utils.model_average import ModelAverage


def set_random_seed(seed, by_rank=False):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    if by_rank:
        seed += get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_trainer(cfg, net_G, net_D=None,
                opt_G=None, opt_D=None,
                sch_G=None, sch_D=None,
                train_data_loader=None,
                val_data_loader=None):
    """Return the trainer object.

    Args:
        cfg (Config): Loaded config object.
        net_G (obj): Generator network object.
        net_D (obj): Discriminator network object.
        opt_G (obj): Generator optimizer object.
        opt_D (obj): Discriminator optimizer object.
        sch_G (obj): Generator optimizer scheduler object.
        sch_D (obj): Discriminator optimizer scheduler object.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.

    Returns:
        (obj): Trainer object.
    """
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, net_G, net_D,
                                  opt_G, opt_D,
                                  sch_G, sch_D,
                                  train_data_loader, val_data_loader)
    return trainer


def get_model_optimizer_and_scheduler(cfg, seed=0):
    r"""Return the networks, the optimizers, and the schedulers. We will
    first set the random seed to a fixed value so that each GPU copy will be
    initialized to have the same network weights. We will then use different
    random seeds for different GPUs. After this we will wrap the generator
    with a moving average model if applicable. It is followed by getting the
    optimizers, amp initialization, and data distributed data parallel wrapping.

    Args:
        cfg (obj): Global configuration.
        seed (int): Random seed.

    Returns:
        (dict):
          - net_G (obj): Generator network object.
          - net_D (obj): Discriminator network object.
          - opt_G (obj): Generator optimizer object.
          - opt_D (obj): Discriminator optimizer object.
          - sch_G (obj): Generator optimizer scheduler object.
          - sch_D (obj): Discriminator optimizer scheduler object.
    """
    # We first set the random seed to be the same so that we initialize each
    # copy of the network in exactly the same way so that they have the same
    # weights and other parameters. The true seed will be the seed.
    set_random_seed(seed, by_rank=False)
    # Construct networks
    lib_G = importlib.import_module(cfg.gen.type)
    lib_D = importlib.import_module(cfg.dis.type)
    net_G = lib_G.Generator(cfg.gen, cfg.data).to('cuda')
    net_D = lib_D.Discriminator(cfg.dis, cfg.data).to('cuda')
    print('Initialize net_G and net_D weights using '
          'type: {} gain: {}'.format(cfg.trainer.init.type,
                                     cfg.trainer.init.gain))
    init_bias = getattr(cfg.trainer.init, 'bias', None)
    net_G.apply(weights_init(
        cfg.trainer.init.type, cfg.trainer.init.gain, init_bias))
    net_D.apply(weights_init(
        cfg.trainer.init.type, cfg.trainer.init.gain, init_bias))
    # Different GPU copies of the same model will receive noises
    # initialized with different random seeds (if applicable) thanks to the
    # set_random_seed command (GPU #K has random seed = args.seed + K).
    set_random_seed(seed, by_rank=True)
    print('net_G parameter count: {:,}'.format(_calculate_model_size(net_G)))
    print('net_D parameter count: {:,}'.format(_calculate_model_size(net_D)))

    # Optimizer
    opt_G = get_optimizer(cfg.gen_opt, net_G)
    opt_D = get_optimizer(cfg.dis_opt, net_D)

    net_G, net_D, opt_G, opt_D = \
        wrap_model_and_optimizer(cfg, net_G, net_D, opt_G, opt_D)

    # Scheduler
    sch_G = get_scheduler(cfg.gen_opt, opt_G)
    sch_D = get_scheduler(cfg.dis_opt, opt_D)

    return net_G, net_D, opt_G, opt_D, sch_G, sch_D


def wrap_model_and_optimizer(cfg, net_G, net_D, opt_G, opt_D):
    r"""Wrap the networks and the optimizers with AMP DDP and (optionally)
    model average.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network object.
        net_D (obj): Discriminator network object.
        opt_G (obj): Generator optimizer object.
        opt_D (obj): Discriminator optimizer object.

    Returns:
        (dict):
          - net_G (obj): Generator network object.
          - net_D (obj): Discriminator network object.
          - opt_G (obj): Generator optimizer object.
          - opt_D (obj): Discriminator optimizer object.
    """
    # Apply model average wrapper.
    if cfg.trainer.model_average:
        net_G = ModelAverage(net_G, cfg.trainer.model_average_beta,
                             cfg.trainer.model_average_start_iteration,
                             cfg.trainer.model_average_remove_sn)
    # AMP initialization.
    [net_G, net_D], [opt_G, opt_D] = \
        amp.initialize([net_G, net_D], [opt_G, opt_D],
                       opt_level=cfg.trainer.amp, num_losses=2)
    # For dealing with numerical issues that might happen during training.
    if cfg.trainer.model_average:
        net_G_module = net_G.module
    else:
        net_G_module = net_G
    if hasattr(net_G_module, 'custom_init'):
        net_G_module.custom_init()

    net_G = _wrap_model(cfg, net_G)
    net_D = _wrap_model(cfg, net_D)
    return net_G, net_D, opt_G, opt_D


def _calculate_model_size(model):
    r"""Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WrappedModel(nn.Module):
    r"""Dummy wrapping the module.
    """

    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        r"""PyTorch module forward function overload."""
        return self.module(*args, **kwargs)


def _wrap_model(cfg, model):
    r"""Wrap a model for apex based distributed data parallel training.

    Args:
        model (obj): PyTorch network model.

    Returns:
        (obj): Wrapped PyTorch network model.
    """
    if torch.distributed.is_available() and dist.is_initialized():
        ddp = cfg.trainer.distributed_data_parallel
        if ddp == 'pytorch':
            return torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True)
        else:
            delay_allreduce = cfg.trainer.delay_allreduce
            return apex.parallel.DistributedDataParallel(
                model, delay_allreduce=delay_allreduce)
    else:
        return WrappedModel(model)


def get_scheduler(cfg_opt, opt):
    """Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if cfg_opt.lr_policy.type == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=cfg_opt.lr_policy.step_size,
            gamma=cfg_opt.lr_policy.gamma)
    elif cfg_opt.lr_policy.type == 'constant':
        scheduler = lr_scheduler.LambdaLR(opt, lambda x: 1)
    else:
        return NotImplementedError('Learning rate policy {} not implemented.'.
                                   format(cfg_opt.lr_policy.type))
    return scheduler


def get_optimizer(cfg_opt, net):
    r"""Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        net (obj): PyTorch network object.

    Returns:
        (obj): Pytorch optimizer
    """
    if hasattr(net, 'get_param_groups'):
        # Allow the network to use different hyper-parameters (e.g., learning
        # rate) for different parameters.
        params = net.get_param_groups(cfg_opt)
    else:
        params = net.parameters()
    return get_optimizer_for_params(cfg_opt, params)


def get_optimizer_for_params(cfg_opt, params):
    r"""Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        params (obj): Parameters to be trained by the parameters.

    Returns:
        (obj): Optimizer
    """
    # We will use fuse optimizers by default.
    fused_opt = cfg_opt.fused_opt
    if cfg_opt.type == 'adam':
        if fused_opt:
            opt = FusedAdam(params,
                            lr=cfg_opt.lr, eps=cfg_opt.eps,
                            betas=(cfg_opt.adam_beta1, cfg_opt.adam_beta2))
        else:
            opt = Adam(params,
                       lr=cfg_opt.lr, eps=cfg_opt.eps,
                       betas=(cfg_opt.adam_beta1, cfg_opt.adam_beta2))

    elif cfg_opt.type == 'madam':
        g_bound = getattr(cfg_opt, 'g_bound', None)
        opt = Madam(params, lr=cfg_opt.lr,
                    scale=cfg_opt.scale, g_bound=g_bound)
    elif cfg_opt.type == 'fromage':
        opt = Fromage(params, lr=cfg_opt.lr)
    elif cfg_opt.type == 'rmsprop':
        opt = RMSprop(params, lr=cfg_opt.lr,
                      eps=cfg_opt.eps, weight_decay=cfg_opt.weight_decay)
    elif cfg_opt.type == 'sgd':
        if fused_opt:
            opt = FusedSGD(params,
                           lr=cfg_opt.lr,
                           momentum=cfg_opt.momentum,
                           weight_decay=cfg_opt.weight_decay)
        else:
            opt = SGD(params,
                      lr=cfg_opt.lr,
                      momentum=cfg_opt.momentum,
                      weight_decay=cfg_opt.weight_decay)
    else:
        raise NotImplementedError(
            'Optimizer {} is not yet implemented.'.format(cfg_opt.type))
    return opt
