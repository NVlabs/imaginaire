# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import argparse

from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_train_and_val_dataloader
from imaginaire.utils.distributed import init_dist
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.logging import init_logging, make_logging_dir
from imaginaire.utils.trainer import (get_model_optimizer_and_scheduler,
                                      get_trainer, set_random_seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config',
                        help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)
    cfg = Config(args.config)

    # If args.single_gpu is set to True,
    # we will disable distributed data parallel
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)
    make_logging_dir(cfg.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    train_data_loader, val_data_loader = get_train_and_val_dataloader(cfg)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D,
                          opt_G, opt_D,
                          sch_G, sch_D,
                          train_data_loader, val_data_loader)
    current_epoch, current_iteration = trainer.load_checkpoint(
        cfg, args.checkpoint,resume=args.resume)

    # Start training.
    for epoch in range(current_epoch, cfg.max_epoch):
        print('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_data_loader.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_data_loader):
            data = trainer.start_of_iteration(data, current_iteration)

            for _ in range(cfg.trainer.dis_step):
                trainer.dis_update(data)
            for _ in range(cfg.trainer.gen_step):
                trainer.gen_update(data)

            current_iteration += 1
            trainer.end_of_iteration(data, current_epoch, current_iteration)
            if current_iteration >= cfg.max_iter:
                print('Done with training!!!')
                return

        current_epoch += 1
        trainer.end_of_epoch(data, current_epoch, current_iteration)
    print('Done with training!!!')
    return


if __name__ == "__main__":
    main()
