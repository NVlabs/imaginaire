# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import math
import os
from functools import partial
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torchvision.models import inception_v3
from cleanfid.features import feature_extractor
from cleanfid.resize import build_resizer

from imaginaire.evaluation.lpips import get_lpips_model
from imaginaire.evaluation.segmentation import get_segmentation_hist_model, get_miou
from imaginaire.evaluation.caption import get_image_encoder, get_r_precision
from imaginaire.evaluation.pretrained import TFInceptionV3, InceptionV3, Vgg16, SwAV
from imaginaire.utils.distributed import (dist_all_gather_tensor, get_rank,
                                          get_world_size, is_master,
                                          is_local_master)
from imaginaire.utils.distributed import master_only_print
from imaginaire.utils.misc import apply_imagenet_normalization, to_cuda


@torch.no_grad()
def compute_all_metrics(act_dir,
                        data_loader,
                        net_G,
                        key_real='images',
                        key_fake='fake_images',
                        sample_size=None,
                        preprocess=None,
                        is_video=False,
                        few_shot_video=False,
                        kid_num_subsets=1,
                        kid_subset_size=None,
                        key_prefix='',
                        prdc_k=5,
                        metrics=None,
                        dataset_name='',
                        aws_credentials=None,
                        **kwargs):
    r"""
    Args:
        act_dir (string): Path to a directory to temporarily save feature activations.
        data_loader (obj): PyTorch dataloader object.
        net_G (obj): The generator module.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        sample_size (int or None): How many samples to use for FID.
        preprocess (func or None): Pre-processing function to use.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
        kid_num_subsets (int): Number of subsets for KID evaluation.
        kid_subset_size (int or None): The number of samples in each subset for KID evaluation.
        key_prefix (string): Add this string before all keys of the output dictionary.
        prdc_k (int): The K used for computing K-NN when evaluating precision/recall/density/coverage.
        metrics (list of strings): Which metrics we want to evaluate.
        dataset_name (string): The name of the dataset, currently only used to determine which segmentation network to
            use for segmentation evaluation.
    Returns:
        batch_y (tensor): Inception features of the current batch. Note that
            only the master gpu will get it.
    """

    from imaginaire.evaluation.fid import _calculate_frechet_distance
    from imaginaire.evaluation.kid import _polynomial_mmd_averages
    from imaginaire.evaluation.prdc import _get_prdc
    from imaginaire.evaluation.msid import _get_msid
    from imaginaire.evaluation.knn import _get_1nn_acc
    if metrics is None:
        metrics = []
    act_path = os.path.join(act_dir, 'activations_real.pt')

    # Get feature activations and other outputs computed from fake images.
    output_module_dict = nn.ModuleDict()
    if "seg_mIOU" in metrics:
        output_module_dict["seg_mIOU"] = get_segmentation_hist_model(dataset_name, aws_credentials)
    if "caption_rprec" in metrics:
        output_module_dict["caption_rprec"] = get_image_encoder(aws_credentials)
    if "LPIPS" in metrics:
        output_module_dict["LPIPS"] = get_lpips_model()

    fake_outputs = get_outputs(
        data_loader, key_real, key_fake, net_G, sample_size, preprocess,
        output_module_dict=output_module_dict, **kwargs
    )
    fake_act = fake_outputs["activations"]

    # Get feature activations computed from real images.
    real_act = load_or_compute_activations(
        act_path, data_loader, key_real, key_fake, None,
        sample_size, preprocess, is_video=is_video,
        few_shot_video=few_shot_video, **kwargs
    )

    metrics_from_activations = {
        "1NN": _get_1nn_acc,
        "MSID": _get_msid,
        "FID": _calculate_frechet_distance,
        "KID": partial(_polynomial_mmd_averages,
                       n_subsets=kid_num_subsets,
                       subset_size=kid_subset_size,
                       ret_var=True),
        "PRDC": partial(_get_prdc, nearest_k=prdc_k)
    }

    other_metrics = {
        "seg_mIOU": get_miou,
        "caption_rprec": get_r_precision,
        "LPIPS": lambda x: {"LPIPS": torch.mean(x).item()}
    }

    all_metrics = {}
    if is_master():
        for metric in metrics:
            if metric in metrics_from_activations:
                metric_function = metrics_from_activations[metric]
                metric_dict = metric_function(real_act, fake_act)
            elif metric in other_metrics:
                metric_function = other_metrics[metric]
                if fake_outputs[metric] is not None:
                    metric_dict = metric_function(fake_outputs[metric])
            else:
                print(f"{metric} is not implemented!")
                raise NotImplementedError
            for k, v in metric_dict.items():
                all_metrics.update({key_prefix + k: v})
    if dist.is_initialized():
        dist.barrier()
    return all_metrics


@torch.no_grad()
def compute_all_metrics_data(data_loader_a,
                             data_loader_b,
                             key_a='images',
                             key_b='images',
                             sample_size=None,
                             preprocess=None,
                             kid_num_subsets=1,
                             kid_subset_size=None,
                             key_prefix='',
                             prdc_k=5,
                             metrics=None,
                             dataset_name='',
                             aws_credentials=None,
                             **kwargs):
    r"""
    Args:
        act_dir (string): Path to a directory to temporarily save feature activations.
        data_loader (obj): PyTorch dataloader object.
        net_G (obj): The generator module.
        key_a (str): Dictionary key value for the real data.
        key_b (str): Dictionary key value for the fake data.
        sample_size (int or None): How many samples to use for FID.
        preprocess (func or None): Pre-processing function to use.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
        kid_num_subsets (int): Number of subsets for KID evaluation.
        kid_subset_size (int or None): The number of samples in each subset for KID evaluation.
        key_prefix (string): Add this string before all keys of the output dictionary.
        prdc_k (int): The K used for computing K-NN when evaluating precision/recall/density/coverage.
        metrics (list of strings): Which metrics we want to evaluate.
        dataset_name (string): The name of the dataset, currently only used to determine which segmentation network to
            use for segmentation evaluation.
    Returns:
        batch_y (tensor): Inception features of the current batch. Note that
            only the master gpu will get it.
    """

    from imaginaire.evaluation.fid import _calculate_frechet_distance
    from imaginaire.evaluation.kid import _polynomial_mmd_averages
    from imaginaire.evaluation.prdc import _get_prdc
    from imaginaire.evaluation.msid import _get_msid
    from imaginaire.evaluation.knn import _get_1nn_acc
    if metrics is None:
        metrics = []

    min_data_size = min(len(data_loader_a.dataset),
                        len(data_loader_b.dataset))
    if sample_size is None:
        sample_size = min_data_size
    else:
        sample_size = min(sample_size, min_data_size)

    # Get feature activations and other outputs computed from fake images.
    output_module_dict = nn.ModuleDict()
    if "seg_mIOU" in metrics:
        output_module_dict["seg_mIOU"] = get_segmentation_hist_model(dataset_name, aws_credentials)
    if "caption_rprec" in metrics:
        output_module_dict["caption_rprec"] = get_image_encoder(aws_credentials)
    if "LPIPS" in metrics:
        output_module_dict["LPIPS"] = get_lpips_model()

    fake_outputs = get_outputs(
        data_loader_b, key_a, key_b, None, sample_size, preprocess,
        output_module_dict=output_module_dict, **kwargs
    )
    act_b = fake_outputs["activations"]

    act_a = load_or_compute_activations(
        None, data_loader_a, key_a, key_b, None, sample_size, preprocess,
        output_module_dict=output_module_dict, **kwargs
    )

    # act_b = load_or_compute_activations(
    #     None, data_loader_b, key_a, key_b, None, sample_size, preprocess,
    #     output_module_dict=output_module_dict, generate_twice=generate_twice, **kwargs
    # )

    metrics_from_activations = {
        "1NN": _get_1nn_acc,
        "MSID": _get_msid,
        "FID": _calculate_frechet_distance,
        "KID": partial(_polynomial_mmd_averages,
                       n_subsets=kid_num_subsets,
                       subset_size=kid_subset_size,
                       ret_var=True),
        "PRDC": partial(_get_prdc, nearest_k=prdc_k)
    }

    other_metrics = {
        "seg_mIOU": get_miou,
        "caption_rprec": get_r_precision,
        "LPIPS": lambda x: {"LPIPS": torch.mean(x).item()}
    }

    all_metrics = {}
    if is_master():
        for metric in metrics:
            if metric in metrics_from_activations:
                metric_function = metrics_from_activations[metric]
                metric_dict = metric_function(act_a, act_b)
            elif metric in other_metrics:
                metric_function = other_metrics[metric]
                if fake_outputs[metric] is not None:
                    metric_dict = metric_function(fake_outputs[metric])
            else:
                print(f"{metric} is not implemented!")
                raise NotImplementedError
            for k, v in metric_dict.items():
                all_metrics.update({key_prefix + k: v})
    if dist.is_initialized():
        dist.barrier()
    return all_metrics


@torch.no_grad()
def get_activations(data_loader, key_real, key_fake,
                    generator=None, sample_size=None, preprocess=None,
                    align_corners=True, network='inception', **kwargs):
    r"""Compute activation values and pack them in a list.

    Args:
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int): How many samples to use for FID.
        preprocess (func): Pre-processing function to use.
        align_corners (bool): The ``'align_corners'`` parameter to be used for
            `torch.nn.functional.interpolate`.
    Returns:
        batch_y (tensor): Inception features of the current batch. Note that
            only the master gpu will get it.
    """
    if dist.is_initialized() and not is_local_master():
        # Make sure only the first process in distributed training downloads
        # the model, and the others will use the cache
        # noinspection PyUnresolvedReferences
        torch.distributed.barrier()

    if network == 'tf_inception':
        model = TFInceptionV3()
    elif network == 'inception':
        model = InceptionV3()
    elif network == 'vgg16':
        model = Vgg16()
    elif network == 'swav':
        model = SwAV()
    elif network == 'clean_inception':
        model = CleanInceptionV3()
    else:
        raise NotImplementedError(f'Network "{network}" is not supported!')

    if dist.is_initialized() and is_local_master():
        # Make sure only the first process in distributed training downloads
        # the model, and the others will use the cache
        # noinspection PyUnresolvedReferences
        dist.barrier()

    model = model.to('cuda').eval()
    world_size = get_world_size()
    batch_y = []

    # Iterate through the dataset to compute the activation.
    for it, data in enumerate(data_loader):
        data = to_cuda(data)
        # Preprocess the data.
        if preprocess is not None:
            data = preprocess(data)
        # Load real data if the generator is not specified.
        if generator is None:
            images = data[key_real]
        else:
            # Compute the generated image.
            net_G_output = generator(data, **kwargs)
            images = net_G_output[key_fake]
        # Clamp the image for models that do not set the output to between
        # -1, 1. For models that employ tanh, this has no effect.
        images.clamp_(-1, 1)
        y = model(images, align_corners=align_corners)
        batch_y.append(y)
        if sample_size is not None and \
                data_loader.batch_size * world_size * (it + 1) >= sample_size:
            # Reach the number of samples we need.
            break

    batch_y = torch.cat(dist_all_gather_tensor(torch.cat(batch_y)))
    if sample_size is not None:
        batch_y = batch_y[:sample_size]
    print(f"Computed feature activations of size {batch_y.shape}")
    return batch_y


class CleanInceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = feature_extractor(name="torchscript_inception", resize_inside=False)

    def forward(self, img_batch, transform=True, **_kwargs):
        if transform:
            # Assume the input is (-1, 1). We transform it to (0, 255) and round it to the closest integer.
            img_batch = torch.round(255 * (0.5 * img_batch + 0.5))
        resized_batch = clean_resize(img_batch)
        return self.model(resized_batch)


def clean_resize(img_batch):
    # Resize images from arbitrary resolutions to 299x299.
    batch_size = img_batch.size(0)
    img_batch = img_batch.cpu().numpy()
    fn_resize = build_resizer('clean')
    resized_batch = torch.zeros(batch_size, 3, 299, 299, device='cuda')
    for idx in range(batch_size):
        curr_img = img_batch[idx]
        img_np = curr_img.transpose((1, 2, 0))
        img_resize = fn_resize(img_np)
        resized_batch[idx] = torch.tensor(img_resize.transpose((2, 0, 1)), device='cuda')
    resized_batch = resized_batch.cuda()
    return resized_batch


@torch.no_grad()
def get_outputs(data_loader, key_real, key_fake,
                generator=None, sample_size=None, preprocess=None,
                align_corners=True, network='inception',
                output_module_dict=None, **kwargs):
    r"""Compute activation values and pack them in a list.

    Args:
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int): How many samples to use for FID.
        preprocess (func): Pre-processing function to use.
        align_corners (bool): The ``'align_corners'`` parameter to be used for `torch.nn.functional.interpolate`.
    Returns:
        batch_y (tensor): Inception features of the current batch. Note that
            only the master gpu will get it.
    """
    if output_module_dict is None:
        output_module_dict = nn.ModuleDict()
    if dist.is_initialized() and not is_local_master():
        # Make sure only the first process in distributed training downloads
        # the model, and the others will use the cache
        # noinspection PyUnresolvedReferences
        torch.distributed.barrier()

    if network == 'tf_inception':
        model = TFInceptionV3()
    elif network == 'inception':
        model = InceptionV3()
    elif network == 'vgg16':
        model = Vgg16()
    elif network == 'swav':
        model = SwAV()
    elif network == 'clean_inception':
        model = CleanInceptionV3()
    else:
        raise NotImplementedError(f'Network "{network}" is not supported!')

    if dist.is_initialized() and is_local_master():
        # Make sure only the first process in distributed training downloads
        # the model, and the others will use the cache
        # noinspection PyUnresolvedReferences
        dist.barrier()

    model = model.to('cuda').eval()
    world_size = get_world_size()
    output = {}
    for k in output_module_dict.keys():
        output[k] = []
    output["activations"] = []

    # Iterate through the dataset to compute the activation.
    for it, data in enumerate(data_loader):
        data = to_cuda(data)
        # Preprocess the data.
        if preprocess is not None:
            data = preprocess(data)
        # Load real data if the generator is not specified.
        if generator is None:
            images = data[key_real]
        else:
            # Compute the generated image.
            net_G_output = generator(data, **kwargs)
            images = net_G_output[key_fake]
        for metric_name, metric_module in output_module_dict.items():
            if metric_module is not None:
                if metric_name == 'LPIPS':
                    assert generator is not None
                    net_G_output_another = generator(data, **kwargs)
                    images_another = net_G_output_another[key_fake]
                    output[metric_name].append(metric_module(images, images_another))
                else:
                    output[metric_name].append(metric_module(data, images, align_corners=align_corners))
        # Clamp the image for models that do not set the output to between
        # -1, 1. For models that employ tanh, this has no effect.
        images.clamp_(-1, 1)
        y = model(images, align_corners=align_corners)
        output["activations"].append(y)
        if sample_size is not None and data_loader.batch_size * world_size * (it + 1) >= sample_size:
            # Reach the number of samples we need.
            break

    for k, v in output.items():
        if len(v) > 0:
            output[k] = torch.cat(dist_all_gather_tensor(torch.cat(v)))[:sample_size]
        else:
            output[k] = None
    return output


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
    inception = inception_init()
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
    master_only_print('Number of videos used for evaluation: {}'.format(num_videos_to_test))
    master_only_print('Number of frames per video used for evaluation: {}'.format(num_frames_per_video))

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
            if few_shot and it == 0:
                continue
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
            y = inception_forward(inception, images)
            batch_y += [y]

    batch_y = torch.cat(batch_y)
    batch_y = dist_all_gather_tensor(batch_y)
    if is_local_master():
        batch_y = torch.cat(batch_y)
    return batch_y


def inception_init():
    inception = inception_v3(pretrained=True, transform_input=False)
    inception = inception.to('cuda')
    inception.eval()
    inception.fc = torch.nn.Sequential()
    return inception


def inception_forward(inception, images):
    images.clamp_(-1, 1)
    images = apply_imagenet_normalization(images)
    images = F.interpolate(images, size=(299, 299),
                           mode='bicubic', align_corners=True)
    return inception(images)


def gather_tensors(batch_y):
    batch_y = torch.cat(batch_y)
    batch_y = dist_all_gather_tensor(batch_y)
    if is_local_master():
        batch_y = torch.cat(batch_y)
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


def load_or_compute_activations(act_path, data_loader, key_real, key_fake,
                                generator=None, sample_size=None,
                                preprocess=None,
                                is_video=False, few_shot_video=False,
                                **kwargs):
    r"""Load mean and covariance from saved npy file if exists. Otherwise,
    compute the mean and covariance.

    Args:
        act_path (str or None): Location for the numpy file to store or to load
            the activations.
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int): How many samples to be used for computing the KID.
        preprocess (func): The preprocess function to be applied to the data.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
    Returns:
        (torch.Tensor) Feature activations.
    """
    if act_path is not None and os.path.exists(act_path):
        # Loading precomputed activations.
        print('Load activations from {}'.format(act_path))
        act = torch.load(act_path, map_location='cpu').cuda()
    else:
        # Compute activations.
        if is_video:
            act = get_video_activations(
                data_loader, key_real, key_fake, generator,
                sample_size, preprocess, few_shot_video, **kwargs
            )
        else:
            act = get_activations(
                data_loader, key_real, key_fake, generator,
                sample_size, preprocess, **kwargs
            )
        if act_path is not None and is_local_master():
            print('Save activations to {}'.format(act_path))
            if not os.path.exists(os.path.dirname(act_path)):
                os.makedirs(os.path.dirname(act_path), exist_ok=True)
            torch.save(act, act_path)
    return act


def compute_pairwise_distance(data_x, data_y=None, num_splits=10):
    r"""

    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    num_samples = data_x.shape[0]
    assert data_x.shape[0] == data_y.shape[0]
    dists = []
    for i in range(num_splits):
        batch_size = math.ceil(num_samples / num_splits)
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        dists.append(torch.cdist(data_x[start_idx:end_idx],
                                 data_y).cpu())
    dists = torch.cat(dists, dim=0)
    return dists


def compute_nn(input_features, k, num_splits=50):
    num_samples = input_features.shape[0]
    all_indices = []
    all_values = []
    for i in range(num_splits):
        batch_size = math.ceil(num_samples / num_splits)
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        dist = torch.cdist(input_features[start_idx:end_idx],
                           input_features)
        dist[:, start_idx:end_idx] += torch.diag(
            float('inf') * torch.ones(dist.size(0), device=dist.device)
        )
        k_smallests, indices = torch.topk(dist, k, dim=-1, largest=False)
        all_indices.append(indices)
        all_values.append(k_smallests)
    return torch.cat(all_values, dim=0), torch.cat(all_indices, dim=0)
