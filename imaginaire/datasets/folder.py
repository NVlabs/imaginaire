# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

from imaginaire.utils.data import IMG_EXTENSIONS


class FolderDataset(data.Dataset):
    r"""This deals with opening, and reading from an Folder dataset.

    Args:
        root (str): Path to the folder.
        metadata (dict): Containing extensions.
    """

    def __init__(self, root, metadata):
        self.root = os.path.expanduser(root)
        self.extensions = metadata

        print('Folder at %s opened.' % (root))

    def getitem_by_path(self, path, data_type):
        r"""Load data item stored for key = path.

        Args:
            path (str): Key into Folder dataset.
            data_type (str): Key into self.extensions e.g. data/data_segmaps/...
        Returns:
            img (PIL.Image) or buf (str): Contents of file for this key.
        """
        # Figure out decoding params.
        ext = self.extensions[data_type]
        if ext in IMG_EXTENSIONS:
            is_image = True
            if 'tif' in ext:
                dtype, mode = np.uint16, -1
            elif 'JPEG' in ext or 'JPG' in ext \
                    or 'jpeg' in ext or 'jpg' in ext:
                dtype, mode = np.uint8, 3
            else:
                dtype, mode = np.uint8, -1
        else:
            is_image = False

        # Get value from key.
        filepath = os.path.join(self.root, path.decode() + '.' + ext)
        assert os.path.exists(filepath), '%s does not exist' % (filepath)
        with open(filepath, 'rb') as f:
            buf = f.read()

        # Decode and return.
        if is_image:
            try:
                img = cv2.imdecode(np.fromstring(buf, dtype=dtype), mode)
            except Exception:
                print(path)
            # BGR to RGB if 3 channels.
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img[:, :, ::-1]
            img = Image.fromarray(img)
            return img
        else:
            return buf

    def __len__(self):
        r"""Return number of keys in Folder dataset."""
        return self.length
