# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import random

from imaginaire.datasets.base import BaseDataset


class Dataset(BaseDataset):
    r"""Unpaired image dataset for use in MUNIT.

    Args:
        cfg (Config): Loaded config object.
        is_inference (bool): In train or inference mode?
    """

    def __init__(self, cfg, is_inference=False, is_test=False):
        super(Dataset, self).__init__(cfg, is_inference, is_test)

    def _create_mapping(self):
        r"""Creates mapping from idx to key in LMDB.

        Returns:
            (tuple):
              - self.mapping (dict): Dict with data type as key mapping idx to
              LMDB key.
              - self.epoch_length (int): Number of samples in an epoch.
        """
        idx_to_key = {}
        for lmdb_idx, sequence_list in enumerate(self.sequence_lists):
            for data_type, data_type_sequence_list in sequence_list.items():
                if data_type not in idx_to_key:
                    idx_to_key[data_type] = []
                for sequence_name, filenames in data_type_sequence_list.items():
                    for filename in filenames:
                        idx_to_key[data_type].append({
                            'lmdb_root': self.lmdb_roots[lmdb_idx],
                            'lmdb_idx': lmdb_idx,
                            'sequence_name': sequence_name,
                            'filename': filename,
                        })
        self.mapping = idx_to_key
        self.epoch_length = max([len(lmdb_keys)
                                 for _, lmdb_keys in self.mapping.items()])
        return self.mapping, self.epoch_length

    def _sample_keys(self, index):
        r"""Gets files to load for this sample.

        Args:
            index (int): Index in [0, len(dataset)].
        Returns:
            keys (dict): Each key of this dict is a data type.
                lmdb_key (dict):
                    lmdb_idx (int): Chosen LMDB dataset root.
                    sequence_name (str): Chosen sequence in chosen dataset.
                    filename (str): Chosen filename in chosen sequence.
        """
        keys = {}
        for data_type in self.data_types:
            lmdb_keys = self.mapping[data_type]
            if self.is_inference:
                # Modulo ensures valid indexing in case A and B have different
                # number of files.
                keys[data_type] = lmdb_keys[index % len(lmdb_keys)]
            else:
                keys[data_type] = random.choice(lmdb_keys)
        return keys

    def __getitem__(self, index):
        r"""Gets selected files.

        Args:
            index (int): Index into dataset.
            concat (bool): Concatenate all items in labels?
        Returns:
            data (dict): Dict with all chosen data_types.
        """
        # Select a sample from the available data.
        keys_per_data_type = self._sample_keys(index)

        # Get keys and lmdbs.
        keys, lmdbs = {}, {}
        for data_type in self.dataset_data_types:
            # Unpack keys.
            lmdb_idx = keys_per_data_type[data_type]['lmdb_idx']
            sequence_name = keys_per_data_type[data_type]['sequence_name']
            filename = keys_per_data_type[data_type]['filename']
            keys[data_type] = '%s/%s' % (sequence_name, filename)
            lmdbs[data_type] = self.lmdbs[data_type][lmdb_idx]

        # Load all data for this index.
        data = self.load_from_dataset(keys, lmdbs)

        # Apply ops pre augmentation.
        data = self.apply_ops(data, self.pre_aug_ops)

        # Do augmentations for images.
        data, is_flipped = self.perform_augmentation(data, paired=False)

        # Apply ops post augmentation.
        data = self.apply_ops(data, self.post_aug_ops)
        data = self.apply_ops(data, self.full_data_post_aug_ops, full_data=True)

        # Convert images to tensor.
        data = self.to_tensor(data)

        # Remove any extra dimensions.
        for data_type in self.image_data_types:
            data[data_type] = data[data_type][0]

        # Package output.
        data['is_flipped'] = is_flipped
        data['key'] = keys_per_data_type

        return data
