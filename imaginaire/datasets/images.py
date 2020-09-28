# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import random

from imaginaire.datasets.base import BaseDataset


class Dataset(BaseDataset):
    r"""Image dataset for use in class conditional GAN.

    Args:
        cfg (Config): Loaded config object.
        is_inference (bool): In train or inference mode?
    """

    def __init__(self, cfg, is_inference=False, is_test=False):
        super(Dataset, self).__init__(cfg, is_inference, is_test)
        self.num_classes = len(self.class_name_to_idx['images'])
        self.sample_class_idx = None

    def set_sample_class_idx(self, class_idx):
        r"""Set sample class idx. This is not used in this class...

        Args:
            class_idx (int): Which class idx to sample from.
        """
        self.sample_class_idx = class_idx
        self.epoch_length = \
            max([len(lmdb_keys) for _, lmdb_keys in self.mapping.items()])

    def _create_mapping(self):
        r"""Creates mapping from idx to key in LMDB.

        Returns:
            (tuple):
              - self.mapping (dict): Dict with data type as key mapping idx to
                LMDB key.
              - self.epoch_length (int): Number of samples in an epoch.
        """
        idx_to_key, class_names = {}, {}
        for lmdb_idx, sequence_list in enumerate(self.sequence_lists):
            for data_type, data_type_sequence_list in sequence_list.items():
                class_names[data_type] = []
                if data_type not in idx_to_key:
                    idx_to_key[data_type] = []
                for sequence_name, filenames in data_type_sequence_list.items():
                    class_name = sequence_name.split('/')[0]
                    for filename in filenames:
                        idx_to_key[data_type].append({
                            'lmdb_root': self.lmdb_roots[lmdb_idx],
                            'lmdb_idx': lmdb_idx,
                            'sequence_name': sequence_name,
                            'filename': filename,
                            'class_name': class_name
                        })
                    class_names[data_type].append(class_name)
        self.mapping = idx_to_key
        self.epoch_length = max([len(lmdb_keys)
                                 for _, lmdb_keys in self.mapping.items()])

        # Create mapping from class name to class idx.
        self.class_name_to_idx = {}
        for data_type, class_names_data_type in class_names.items():
            self.class_name_to_idx[data_type] = {}
            class_names_data_type = sorted(list(set(class_names_data_type)))
            for class_idx, class_name in enumerate(class_names_data_type):
                self.class_name_to_idx[data_type][class_name] = class_idx

        # Add class idx to mapping.
        for data_type in self.mapping:
            for key in self.mapping[data_type]:
                key['class_idx'] = \
                    self.class_name_to_idx[data_type][key['class_name']]

        # Create a mapping from index to lmdb key for each class.
        idx_to_key_class = {}
        for data_type in self.mapping:
            idx_to_key_class[data_type] = {}
            for class_idx, class_name in enumerate(class_names[data_type]):
                idx_to_key_class[data_type][class_idx] = []
            for key in self.mapping[data_type]:
                idx_to_key_class[data_type][key['class_idx']].append(key)
        self.mapping_class = idx_to_key_class

        return self.mapping, self.epoch_length

    def _sample_keys(self, index):
        r"""Gets files to load for this sample.

        Args:
            index (int): Index in [0, len(dataset)].
        Returns:
            keys (dict): Each key of this dict is a data type.
              - lmdb_key (dict):
                  - lmdb_idx (int): Chosen LMDB dataset root.
                  - sequence_name (str): Chosen sequence in chosen dataset.
                  - filename (str): Chosen filename in chosen sequence.
        """

        keys = {}
        if self.is_inference:  # evaluation mode
            lmdb_keys = self.mapping['images']
            keys['images'] = lmdb_keys[index % len(lmdb_keys)]
        else:
            lmdb_keys = self.mapping['images']
            keys['images'] = random.choice(lmdb_keys)
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

        # Get class idx into a list.
        class_idxs = []
        for data_type in keys_per_data_type:
            class_idxs.append(keys_per_data_type[data_type]['class_idx'])

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
        data['labels'] = class_idxs[0]
        return data
