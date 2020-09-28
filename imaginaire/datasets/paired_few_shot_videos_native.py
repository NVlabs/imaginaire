# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import copy
import random
import tempfile
from collections import OrderedDict

import numpy as np
import torch
import torchvision.io as io
from PIL import Image

from imaginaire.datasets.base import BaseDataset


class Dataset(BaseDataset):
    r"""Dataset for paired few shot videos.

    Args:
        cfg (Config): Loaded config object.
        is_inference (bool): In train or inference mode?
    """

    def __init__(self, cfg, is_inference=False, is_test=False):
        super(Dataset, self).__init__(cfg, is_inference, is_test)
        self.is_video_dataset = True
        if hasattr(cfg.data, 'first_last_only'):
            self.first_last_only = cfg.data.first_last_only
        else:
            self.first_last_only = False

    def get_label_lengths(self):
        r"""Get num channels of all labels to be concated.

        Returns:
            label_lengths (OrderedDict): Dict mapping image data_type to num
            channels.
        """
        label_lengths = OrderedDict()
        for data_type in self.input_labels:
            label_lengths[data_type] = self.num_channels[data_type]
        return label_lengths

    def num_inference_sequences(self):
        r"""Number of sequences available for inference.

        Returns:
           (int)
        """
        assert self.is_inference
        return len(self.mapping)

    def _create_mapping(self):
        r"""Creates mapping from idx to key in LMDB.

        Returns:
            (tuple):
              - self.mapping (dict): Dict of seq_len to list of sequences.
              - self.epoch_length (int): Number of samples in an epoch.
        """
        # Create dict mapping length to sequence.
        mapping = []
        for lmdb_idx, sequence_list in enumerate(self.sequence_lists):
            for sequence_name, filenames in sequence_list.items():
                for filename in filenames:
                    # This file is corrupt.
                    if filename == 'z-KziTO_5so_0019_start0_end85_h596_w596':
                        continue
                    mapping.append({
                        'lmdb_root': self.lmdb_roots[lmdb_idx],
                        'lmdb_idx': lmdb_idx,
                        'sequence_name': sequence_name,
                        'filenames': [filename],
                    })
        self.mapping = mapping
        self.epoch_length = len(mapping)

        return self.mapping, self.epoch_length

    def _sample_keys(self, index):
        r"""Gets files to load for this sample.

        Args:
            index (int): Index in [0, len(dataset)].
        Returns:
            (tuple):
              - key (dict):
                - lmdb_idx (int): Chosen LMDB dataset root.
                - sequence_name (str): Chosen sequence in chosen dataset.
                - filenames (list of str): Chosen filenames in chosen sequence.
        """
        if self.is_inference:
            assert index < self.epoch_length
            raise NotImplementedError
        else:
            # Select a video at random.
            key = random.choice(self.mapping)
        return key

    def _create_sequence_keys(self, sequence_name, filenames):
        r"""Create the LMDB key for this piece of information.

        Args:
            sequence_name (str): Which sequence from the chosen dataset.
            filenames (list of str): List of filenames in this sequence.
        Returns:
            keys (list): List of full keys.
        """
        assert isinstance(filenames, list), 'Filenames should be a list.'
        keys = []
        for filename in filenames:
            keys.append('%s/%s' % (sequence_name, filename))
        return keys

    def _getitem(self, index, concat=True):
        r"""Gets selected files.

        Args:
            index (int): Index into dataset.
            concat (bool): Concatenate all items in labels?
        Returns:
            data (dict): Dict with all chosen data_types.
        """
        # Select a sample from the available data.
        keys = self._sample_keys(index)

        # Unpack keys.
        lmdb_idx = keys['lmdb_idx']
        sequence_name = keys['sequence_name']
        filenames = keys['filenames']

        # Get key and lmdbs.
        keys, lmdbs = {}, {}
        for data_type in self.dataset_data_types:
            keys[data_type] = self._create_sequence_keys(
                sequence_name, filenames)
            lmdbs[data_type] = self.lmdbs[data_type][lmdb_idx]

        # Load all data for this index.
        data = self.load_from_dataset(keys, lmdbs)

        # Get frames from video.
        temp = tempfile.NamedTemporaryFile()
        temp.write(data['videos'][0])
        temp.seek(0)

        try:
            frames, _, info = io.read_video(temp)
            if self.first_last_only:
                chosen_idxs = [0, frames.size(0) - 1]
            else:
                chosen_idxs = random.sample(range(frames.size(0)), 2)
            chosen_images = []
            for idx in chosen_idxs:
                chosen_images.append(Image.fromarray(frames[idx].numpy()))
        except Exception:
            print('Issue with file:', sequence_name, filenames)
            blank = np.zeros((512, 512, 3), dtype=np.uint8)
            chosen_images = [Image.fromarray(blank), Image.fromarray(blank)]

        data['videos'] = chosen_images

        # Apply ops pre augmentation.
        data = self.apply_ops(data, self.pre_aug_ops)

        # Do augmentations for images.
        data, is_flipped = self.perform_augmentation(data, paired=True)

        # Create copy of keypoint data types before post aug.
        kp_data = {}
        for data_type in self.keypoint_data_types:
            new_key = data_type + '_xy'
            kp_data[new_key] = copy.deepcopy(data[data_type])

        # Apply ops post augmentation.
        data = self.apply_ops(data, self.post_aug_ops)

        # Convert images to tensor.
        data = self.to_tensor(data)

        # Do one-hot encoding of required image labels.
        data = self.make_one_hot(data)

        # Pack the sequence of images.
        for data_type in self.image_data_types:
            for idx in range(len(data[data_type])):
                data[data_type][idx] = data[data_type][idx].unsqueeze(0)
            data[data_type] = torch.cat(data[data_type], dim=0)

        # Package output.
        if concat and self.input_labels:
            labels = []
            for data_type in self.input_labels:
                label = data.pop(data_type)
                labels.append(label)
            data['label'] = torch.cat(labels, dim=1)
            if not self.is_video_dataset:
                data['label'] = data['label'].squeeze(0)

        if not self.is_video_dataset:
            # Remove any extra dimensions.
            for data_type in self.image_data_types:
                if data_type in data:
                    data[data_type] = data[data_type].squeeze(0)

        # Add keypoint xy to data.
        data.update(kp_data)

        # Prepare output.
        data['driving_images'] = data['videos'][0]
        data['source_images'] = data['videos'][1]
        data.pop('videos')
        data['is_flipped'] = is_flipped
        data['key'] = keys
        data['original_h_w'] = torch.IntTensor([
            self.augmentor.original_h, self.augmentor.original_w])

        # Apply full data ops.
        data = self.apply_ops(data, self.full_data_ops, full_data=True)

        return data

    def __getitem__(self, index):
        return self._getitem(index, concat=True)
