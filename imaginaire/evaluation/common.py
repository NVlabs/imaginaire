# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch.nn import functional as F
from torchvision.models import inception_v3

from imaginaire.utils.distributed import (dist_all_gather_tensor, get_rank,
                                          get_world_size, is_master)
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.misc import apply_imagenet_normalization, to_cuda


@torch.no_grad()
def get_activations(data_loader, key_real, key_fake,
                    generator=None, sample_size=None, preprocess=None):
    r"""Compute activation values and pack them in a list.

    Args:
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int): How many samples to use for FID.
        preprocess (func): Pre-processing function to use.
    Returns:
        batch_y (tensor): Inception features of the current batch. Note that
            only the master gpu will get it.
    """
    # Load pretrained inception_v3 network and set it in GPU evaluation mode.
    inception = inception_v3(pretrained=True, transform_input=False,
                             init_weights=False)
    inception = inception.to('cuda').eval()

    # Disable the fully connected layer in the output.
    inception.fc = torch.nn.Sequential()

    world_size = get_world_size()
    batch_y = []
    # Iterate through the dataset to compute the activation.
    for it, data in enumerate(data_loader):
        data = to_cuda(data)
        # preprocess the data is preprocess is not none.
        if preprocess is not None:
            data = preprocess(data)
        # Load real data if trainer is not specified.
        if generator is None:
            images = data[key_real]
        else:
            # Compute the generated image.
            net_G_output = generator(data)
            images = net_G_output[key_fake]
        # Clamp the image for models that do not set the output to between
        # -1, 1. For models that employ tanh, this has no effect.
        images.clamp_(-1, 1)
        images = apply_imagenet_normalization(images)
        images = F.interpolate(images, size=(299, 299),
                               mode='bilinear', align_corners=True)
        y = inception(images)
        batch_y.append(y)
        if sample_size is not None and \
                data_loader.batch_size * world_size * (it + 1) >= sample_size:
            # Reach the number of samples we need.
            break

    batch_y = torch.cat(batch_y)
    batch_y = dist_all_gather_tensor(batch_y)
    if is_master():
        batch_y = torch.cat(batch_y).cpu().data.numpy()
        if sample_size is not None:
            batch_y = batch_y[:sample_size]
        print(batch_y.shape)
        return batch_y
    else:
        return None


@torch.no_grad()
def get_video_activations(data_loader, key_real, key_fake, trainer=None,
                          sample_size=None, preprocess=None, few_shot=False):
    r"""Compute activation values and pack them in a list. We do not do all
    reduce here.

    Args:
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        trainer (obj): Trainer. Video generation is more involved, we rely on
            the "reset" and "test" function to conduct the evaluation.
        sample_size (int): For computing video activation, we will use .
        preprocess (func): The preprocess function to be applied to the data.
        few_shot (bool): If ``True``, uses the few-shot setting.
    Returns:
        batch_y (tensor): Inception features of the current batch. Note that
            only the master gpu will get it.
    """
    inception = inception_v3(pretrained=True, transform_input=False)
    inception = inception.to('cuda')
    inception.eval()
    inception.fc = torch.nn.Sequential()
    batch_y = []

    # We divide video sequences to different GPUs for testing.
    num_sequences = data_loader.dataset.num_inference_sequences()
    if sample_size is None:
        num_videos_to_test = 10
        num_frames_per_video = 5
    else:
        num_videos_to_test, num_frames_per_video = sample_size
    if num_videos_to_test == -1:
        num_videos_to_test = num_sequences
    else:
        num_videos_to_test = min(num_videos_to_test, num_sequences)
    print('Number of videos used for evaluation: {}'.format(
        num_videos_to_test))
    print('Number of frames per video used for evaluation: {}'.format(
        num_frames_per_video))

    world_size = get_world_size()
    if num_videos_to_test < world_size:
        seq_to_run = [get_rank() % num_videos_to_test]
    else:
        num_videos_to_test = num_videos_to_test // world_size * world_size
        seq_to_run = range(get_rank(), num_videos_to_test, world_size)

    for sequence_idx in seq_to_run:
        data_loader = set_sequence_idx(few_shot, data_loader, sequence_idx)
        if trainer is not None:
            trainer.reset()
        for it, data in enumerate(data_loader):
            if it >= num_frames_per_video:
                break

            # preprocess the data is preprocess is not none.
            if trainer is not None:
                data = trainer.pre_process(data)
            elif preprocess is not None:
                data = preprocess(data)
            data = to_cuda(data)

            if trainer is None:
                images = data[key_real][:, -1]
            else:
                net_G_output = trainer.test_single(data)
                images = net_G_output[key_fake]
            images.clamp_(-1, 1)
            images = apply_imagenet_normalization(images)
            images = F.interpolate(images, size=(299, 299),
                                   mode='bilinear', align_corners=True)
            y = inception(images)
            batch_y += [y]

    batch_y = torch.cat(batch_y)
    batch_y = dist_all_gather_tensor(batch_y)
    if is_master():
        batch_y = torch.cat(batch_y).cpu().data.numpy()
    return batch_y


def set_sequence_idx(few_shot, data_loader, sequence_idx):
    r"""Get sequence index

    Args:
        few_shot (bool): If ``True``, uses the few-shot setting.
        data_loader: dataloader object
        sequence_idx (int): which sequence to use.
    """
    if few_shot:
        data_loader.dataset.set_inference_sequence_idx(sequence_idx,
                                                       sequence_idx,
                                                       0)
    else:
        data_loader.dataset.set_inference_sequence_idx(sequence_idx)
    return data_loader
