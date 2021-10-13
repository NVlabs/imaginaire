# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import argparse
import os
import sys
import random

import torch.autograd.profiler as profiler
import wandb

import imaginaire.config
from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_train_and_val_dataloader
from imaginaire.utils.distributed import init_dist, is_master, get_world_size
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.misc import slice_tensor
from imaginaire.utils.logging import init_logging, make_logging_dir
from imaginaire.utils.trainer import (get_model_optimizer_and_scheduler,
                                      get_trainer, set_random_seed)

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config',
                        help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--randomized_seed', action='store_true', help='Use a random seed between 0-10000.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_jit', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--wandb_id', type=str)
    parser.add_argument('--resume', type=int)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    if args.randomized_seed:
        args.seed = random.randint(0, 10000)
    set_random_seed(args.seed, by_rank=True)
    cfg = Config(args.config)
    try:
        from userlib.auto_resume import AutoResume
        AutoResume.init()
    except:  # noqa
        pass

    # If args.single_gpu is set to True,
    # we will disable distributed data parallel
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)
    print(f"Training with {get_world_size()} GPUs.")

    # Global arguments.
    imaginaire.config.DEBUG = args.debug
    imaginaire.config.USE_JIT = args.use_jit

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)
    make_logging_dir(cfg.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    batch_size = cfg.data.train.batch_size
    total_step = max(cfg.trainer.dis_step, cfg.trainer.gen_step)
    cfg.data.train.batch_size *= total_step
    train_data_loader, val_data_loader = get_train_and_val_dataloader(cfg, args.seed)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D,
                          opt_G, opt_D,
                          sch_G, sch_D,
                          train_data_loader, val_data_loader)
    resumed, current_epoch, current_iteration = trainer.load_checkpoint(cfg, args.checkpoint, args.resume)

    # Initialize Wandb.
    if is_master():
        if args.wandb_id is not None:
            wandb_id = args.wandb_id
        else:
            if resumed and os.path.exists(os.path.join(cfg.logdir, 'wandb_id.txt')):
                with open(os.path.join(cfg.logdir, 'wandb_id.txt'), 'r+') as f:
                    wandb_id = f.read()
            else:
                wandb_id = wandb.util.generate_id()
                with open(os.path.join(cfg.logdir, 'wandb_id.txt'), 'w+') as f:
                    f.write(wandb_id)
        wandb_mode = "disabled" if (args.debug or not args.wandb) else "online"
        wandb.init(id=wandb_id,
                   project=args.wandb_name,
                   config=cfg,
                   name=os.path.basename(cfg.logdir),
                   resume="allow",
                   settings=wandb.Settings(start_method="fork"),
                   mode=wandb_mode)
        wandb.config.update({'dataset': cfg.data.name})
        wandb.watch(trainer.net_G_module)
        wandb.watch(trainer.net_D.module)

    # Start training.
    for epoch in range(current_epoch, cfg.max_epoch):
        print('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_data_loader.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_data_loader):
            with profiler.profile(enabled=args.profile,
                                  use_cuda=True,
                                  profile_memory=True,
                                  record_shapes=True) as prof:
                data = trainer.start_of_iteration(data, current_iteration)

                for i in range(cfg.trainer.dis_step):
                    trainer.dis_update(
                        slice_tensor(data, i * batch_size,
                                     (i + 1) * batch_size))
                for i in range(cfg.trainer.gen_step):
                    trainer.gen_update(
                        slice_tensor(data, i * batch_size,
                                     (i + 1) * batch_size))

                current_iteration += 1
                trainer.end_of_iteration(data, current_epoch, current_iteration)
                if current_iteration >= cfg.max_iter:
                    print('Done with training!!!')
                    return
            if args.profile:
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                prof.export_chrome_trace(os.path.join(cfg.logdir, "trace.json"))
            try:
                if AutoResume.termination_requested():
                    trainer.save_checkpoint(current_epoch, current_iteration)
                    AutoResume.request_resume()
                    print("Training terminated. Returning")
                    return 0
            except:  # noqa
                pass

        current_epoch += 1
        trainer.end_of_epoch(data, current_epoch, current_iteration)
    print('Done with training!!!')
    return


if __name__ == "__main__":
    main()
