# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import requests
import torch.distributed as dist
import torchvision.utils

from imaginaire.utils.distributed import is_master


def save_pilimage_in_jpeg(fullname, output_img):
    r"""Save PIL Image to JPEG.

    Args:
        fullname (str): Full save path.
        output_img (PIL Image): Image to be saved.
    """
    dirname = os.path.dirname(fullname)
    os.makedirs(dirname, exist_ok=True)
    output_img.save(fullname, 'JPEG', quality=99)


def save_intermediate_training_results(
        visualization_images, logdir, current_epoch, current_iteration):
    r"""Save intermediate training results for debugging purpose.

    Args:
        visualization_images (tensor): Image where pixel values are in [-1, 1].
        logdir (str): Where to save the image.
        current_epoch (int): Current training epoch.
        current_iteration (int): Current training iteration.
    """
    visualization_images = (visualization_images + 1) / 2
    output_filename = os.path.join(
        logdir, 'images',
        'epoch_{:05}iteration{:09}.jpg'.format(
            current_epoch, current_iteration))
    print('Save output images to {}'.format(output_filename))
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    image_grid = torchvision.utils.make_grid(
        visualization_images.data, nrow=1, padding=0, normalize=False)
    torchvision.utils.save_image(image_grid, output_filename, nrow=1)


def download_file_from_google_drive(URL, destination):
    r"""Download a file from google drive.

    Args:
        URL: GDrive file ID.
        destination: Path to save the file.

    Returns:

    """
    download_file(f"https://docs.google.com/uc?export=download&id={URL}", destination)


def download_file(URL, destination):
    r"""Download a file from google drive or pbss by using the url.

    Args:
        URL: GDrive URL or PBSS pre-signed URL for the checkpoint.
        destination: Path to save the file.

    Returns:

    """
    session = requests.Session()
    response = session.get(URL, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    r"""Get confirm token

    Args:
        response: Check if the file exists.

    Returns:

    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    r"""Save response content

    Args:
        response:
        destination: Path to save the file.

    Returns:

    """
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def get_checkpoint(checkpoint_path, url=''):
    r"""Get the checkpoint path. If it does not exist yet, download it from
    the url.

    Args:
        checkpoint_path (str): Checkpoint path.
        url (str): URL to download checkpoint.
    Returns:
        (str): Full checkpoint path.
    """
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = os.getcwd()
    save_dir = os.path.join(os.environ['TORCH_HOME'], 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    full_checkpoint_path = os.path.join(save_dir, checkpoint_path)
    if not os.path.exists(full_checkpoint_path):
        os.makedirs(os.path.dirname(full_checkpoint_path), exist_ok=True)
        if is_master():
            print('Downloading {}'.format(url))
            if 'pbss.s8k.io' not in url:
                url = f"https://docs.google.com/uc?export=download&id={url}"
            download_file(url, full_checkpoint_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return full_checkpoint_path
