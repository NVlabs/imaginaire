# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import json
import os

import cv2
import lmdb
import numpy as np
import torch.utils.data as data
from PIL import Image

from imaginaire.utils.data import IMG_EXTENSIONS


class LMDBDataset(data.Dataset):
    r"""This deals with opening, and reading from an LMDB dataset.
    Args:
        root (str): Path to the LMDB file.
    """

    def __init__(self, root):
        self.root = os.path.expanduser(root)
        self.env = lmdb.open(root, max_readers=126, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        # Read metadata.
        with open(os.path.join(self.root, '..', 'metadata.json')) as fin:
            self.extensions = json.load(fin)

        print('LMDB file at %s opened.' % (root))

    def getitem_by_path(self, path, data_type):
        r"""Load data item stored for key = path.

        Args:
            path (str): Key into LMDB dataset.
            data_type (str): Key into self.extensions e.g. data/data_segmaps/...
        Returns:
            img (PIL.Image) or buf (str): Contents of LMDB value for this key.
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
        with self.env.begin(write=False) as txn:
            buf = txn.get(path)

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
        r"""Return number of keys in LMDB dataset."""
        return self.length
