# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import io
import json

# import cv2
import boto3
from botocore.config import Config
import numpy as np
import torch.utils.data as data
from PIL import Image
import imageio
from botocore.exceptions import ClientError

from imaginaire.datasets.cache import Cache
from imaginaire.utils.data import IMG_EXTENSIONS, HDR_IMG_EXTENSIONS

Image.MAX_IMAGE_PIXELS = None


class ObjectStoreDataset(data.Dataset):
    r"""This deals with opening, and reading from an AWS S3 bucket.
    Args:

        root (str): Path to the AWS S3 bucket.
        aws_credentials_file (str): Path to file containing AWS credentials.
        data_type (str): Which data type should this dataset load?
    """

    def __init__(self, root, aws_credentials_file, data_type='', cache=None):
        # Cache.
        self.cache = False
        if cache is not None:
            # raise NotImplementedError
            self.cache = Cache(cache.root, cache.size_GB)

        # Get bucket info, and keys to info about dataset.
        with open(aws_credentials_file) as fin:
            self.credentials = json.load(fin)

        parts = root.split('/')
        self.bucket = parts[0]
        self.all_filenames_key = '/'.join(parts[1:]) + '/all_filenames.json'
        self.metadata_key = '/'.join(parts[1:]) + '/metadata.json'

        # Get list of filenames.
        filename_info = self._get_object(self.all_filenames_key)
        self.sequence_list = json.loads(filename_info.decode('utf-8'))

        # Get length.
        length = 0
        for _, value in self.sequence_list.items():
            length += len(value)
        self.length = length

        # Read metadata.
        metadata_info = self._get_object(self.metadata_key)
        self.extensions = json.loads(metadata_info.decode('utf-8'))
        self.data_type = data_type

        print('AWS S3 bucket at %s opened.' % (root + '/' + self.data_type))

    def _get_object(self, key):
        r"""Download object from bucket.

        Args:
            key (str): Key inside bucket.
        """
        # Look up value in cache.
        object_content = self.cache.read(key) if self.cache else False
        if not object_content:
            # Either no cache used or key not found in cache.
            config = Config(connect_timeout=30,
                            signature_version="s3",
                            retries={"max_attempts": 999999})
            s3 = boto3.client('s3', **self.credentials, config=config)
            try:
                s3_response_object = s3.get_object(Bucket=self.bucket, Key=key)
                object_content = s3_response_object['Body'].read()
            except Exception as e:
                print('%s not found' % (key))
                print(e)
            # Save content to cache.
            if self.cache:
                self.cache.write(key, object_content)
        return object_content

    def getitem_by_path(self, path, data_type):
        r"""Load data item stored for key = path.

        Args:
            path (str): Path into AWS S3 bucket, without data_type prefix.
            data_type (str): Key into self.extensions e.g. data/data_segmaps/...
        Returns:
            img (PIL.Image) or buf (str): Contents of LMDB value for this key.
        """
        # Figure out decoding params.
        ext = self.extensions[data_type]
        is_image = False
        is_hdr = False
        parts = path.split('/')
        key = parts[0] + '/' + data_type + '/' + '/'.join(parts[1:]) + '.' + ext
        if ext in IMG_EXTENSIONS:
            is_image = True
            if 'tif' in ext:
                _, mode = np.uint16, -1
            elif 'JPEG' in ext or 'JPG' in ext \
                    or 'jpeg' in ext or 'jpg' in ext:
                _, mode = np.uint8, 3
            else:
                _, mode = np.uint8, -1
        elif ext in HDR_IMG_EXTENSIONS:
            is_hdr = True
        else:
            is_image = False

        # Get value from key.
        buf = self._get_object(key)

        # Decode and return.
        if is_image:
            # This is totally a hack.
            # We should have a better way to handle grayscale images.
            img = Image.open(io.BytesIO(buf))
            if mode == 3:
                img = img.convert('RGB')
            return img
        elif is_hdr:
            try:
                imageio.plugins.freeimage.download()
                img = imageio.imread(buf)
            except Exception:
                print(path)
            return img  # Return a numpy array
        else:
            return buf

    def __len__(self):
        r"""Return number of keys in LMDB dataset."""
        return self.length
