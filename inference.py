# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import argparse

from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_test_dataloader
from imaginaire.utils.distributed import init_dist
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.io import get_checkpoint
from imaginaire.utils.logging import init_logging
from imaginaire.utils.trainer import \
    (get_model_optimizer_and_scheduler, get_trainer, set_random_seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', required=True,
                        help='Path to the training config file.')
    parser.add_argument('--checkpoint', default='',
                        help='Checkpoint path.')
    parser.add_argument('--output_dir', required=True,
                        help='Location to save the image outputs')
    parser.add_argument('--logdir',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)
    cfg = Config(args.config)
    if not hasattr(cfg, 'inference_args'):
        cfg.inference_args = None

    # If args.single_gpu is set to True,
    # we will disable distributed data parallel.
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    test_data_loader = get_test_dataloader(cfg)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D,
                          opt_G, opt_D,
                          sch_G, sch_D,
                          None, test_data_loader)

    if args.checkpoint == '':
        # Download pretrained weights.
        pretrained_weight_url = cfg.pretrained_weight
        if pretrained_weight_url == '':
            print('google link to the pretrained weight is not specified.')
            raise
        default_checkpoint_path = args.config.replace('.yaml', '.pt')
        args.checkpoint = get_checkpoint(
            default_checkpoint_path, pretrained_weight_url)
        print('Checkpoint downloaded to', args.checkpoint)

    # Load checkpoint.
    trainer.load_checkpoint(cfg, args.checkpoint)

    # Do inference.
    trainer.current_epoch = -1
    trainer.current_iteration = -1
    trainer.test(test_data_loader, args.output_dir, cfg.inference_args)


if __name__ == "__main__":
    main()
