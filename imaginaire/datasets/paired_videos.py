# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import copy
import random
from collections import OrderedDict

import torch

from imaginaire.datasets.base import BaseDataset
from imaginaire.model_utils.fs_vid2vid import select_object
from imaginaire.utils.distributed import master_only_print as print


class Dataset(BaseDataset):
    r"""Paired video dataset for use in vid2vid, wc_vid2vid.

    Args:
        cfg (Config): Loaded config object.
        is_inference (bool): In train or inference mode?
        sequence_length (int): What sequence of images to provide?
    """

    def __init__(self, cfg,
                 is_inference=False,
                 sequence_length=None,
                 is_test=False):
        # Get initial sequence length.
        if sequence_length is None and not is_inference:
            self.sequence_length = cfg.data.train.initial_sequence_length
        elif sequence_length is None and is_inference:
            self.sequence_length = 2
        else:
            self.sequence_length = sequence_length
        super(Dataset, self).__init__(cfg, is_inference, is_test)
        self.set_sequence_length(self.sequence_length)
        self.is_video_dataset = True

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

    def set_inference_sequence_idx(self, index):
        r"""Get frames from this sequence during inference.

        Args:
            index (int): Index of inference sequence.
        """
        assert self.is_inference
        assert index < len(self.mapping)
        self.inference_sequence_idx = index
        self.epoch_length = len(
            self.mapping[self.inference_sequence_idx]['filenames'])

    def set_sequence_length(self, sequence_length):
        r"""Set the length of sequence you want as output from dataloader.

        Args:
            sequence_length (int): Length of output sequences.
        """
        assert isinstance(sequence_length, int)
        if sequence_length > self.sequence_length_max:
            print('Requested sequence length (%d) > ' % (sequence_length) +
                  'max sequence length (%d). ' % (self.sequence_length_max) +
                  'Limiting sequence length to max sequence length.')
            sequence_length = self.sequence_length_max
        self.sequence_length = sequence_length
        # Recalculate mapping as some sequences might no longer be useful.
        self.mapping, self.epoch_length = self._create_mapping()
        print('Epoch length:', self.epoch_length)

    def _compute_dataset_stats(self):
        r"""Compute statistics of video sequence dataset.

        Returns:
            sequence_length_max (int): Maximum sequence length.
        """
        print('Num datasets:', len(self.sequence_lists))

        if self.sequence_length >= 1:
            num_sequences, sequence_length_max = 0, 0
            for sequence in self.sequence_lists:
                for _, filenames in sequence.items():
                    sequence_length_max = max(
                        sequence_length_max, len(filenames))
                    num_sequences += 1
            print('Num sequences:', num_sequences)
            print('Max sequence length:', sequence_length_max)
            self.sequence_length_max = sequence_length_max

    def _create_mapping(self):
        r"""Creates mapping from idx to key in LMDB.

        Returns:
            (tuple):
              - self.mapping (dict): Dict of seq_len to list of sequences.
              - self.epoch_length (int): Number of samples in an epoch.
        """
        # Create dict mapping length to sequence.
        length_to_key, num_selected_seq = {}, 0
        total_num_of_frames = 0
        for lmdb_idx, sequence_list in enumerate(self.sequence_lists):
            for sequence_name, filenames in sequence_list.items():
                if len(filenames) >= self.sequence_length:
                    total_num_of_frames += len(filenames)
                    if len(filenames) not in length_to_key:
                        length_to_key[len(filenames)] = []
                    length_to_key[len(filenames)].append({
                        'lmdb_root': self.lmdb_roots[lmdb_idx],
                        'lmdb_idx': lmdb_idx,
                        'sequence_name': sequence_name,
                        'filenames': filenames,
                    })
                    num_selected_seq += 1
        self.mapping = length_to_key
        self.epoch_length = num_selected_seq
        if not self.is_inference and self.epoch_length < \
                self.cfgdata.train.batch_size * 8:
            self.epoch_length = total_num_of_frames

        # At inference time, we want to use all sequences,
        # irrespective of length.
        if self.is_inference:
            sequence_list = []
            for key, sequences in self.mapping.items():
                sequence_list.extend(sequences)
            self.mapping = sequence_list

        return self.mapping, self.epoch_length

    def _sample_keys(self, index):
        r"""Gets files to load for this sample.

        Args:
            index (int): Index in [0, len(dataset)].
        Returns:
            key (dict):
              - lmdb_idx (int): Chosen LMDB dataset root.
              - sequence_name (str): Chosen sequence in chosen dataset.
              - filenames (list of str): Chosen filenames in chosen sequence.
        """
        if self.is_inference:
            assert index < self.epoch_length
            chosen_sequence = self.mapping[self.inference_sequence_idx]
            chosen_filenames = [chosen_sequence['filenames'][index]]
        else:
            # Pick a time step for temporal augmentation.
            time_step = random.randint(1, self.augmentor.max_time_step)
            required_sequence_length = 1 + \
                (self.sequence_length - 1) * time_step

            # If step is too large, default to step size of 1.
            if required_sequence_length > self.sequence_length_max:
                required_sequence_length = self.sequence_length
                time_step = 1

            # Find valid sequences.
            valid_sequences = []
            for sequence_length, sequences in self.mapping.items():
                if sequence_length >= required_sequence_length:
                    valid_sequences.extend(sequences)

            # Pick a sequence.
            chosen_sequence = random.choice(valid_sequences)

            # Choose filenames.
            max_start_idx = len(chosen_sequence['filenames']) - \
                required_sequence_length
            start_idx = random.randint(0, max_start_idx)

            chosen_filenames = chosen_sequence['filenames'][
                start_idx:start_idx + required_sequence_length:time_step]
            assert len(chosen_filenames) == self.sequence_length

        # Prepre output key.
        key = copy.deepcopy(chosen_sequence)
        key['filenames'] = chosen_filenames
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
        if sequence_name.endswith('___') and sequence_name[-9:-6] == '___':
            sequence_name = sequence_name[:-9]
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

        # Apply ops pre augmentation.
        data = self.apply_ops(data, self.pre_aug_ops)

        # If multiple subjects exist in the data, only pick one to synthesize.
        data = select_object(data, obj_indices=None)

        # Do augmentations for images.
        data, is_flipped = self.perform_augmentation(data, paired=True)

        # Create copy of keypoint data types before post aug.
        kp_data = {}
        for data_type in self.keypoint_data_types:
            new_key = data_type + '_xy'
            kp_data[new_key] = copy.deepcopy(data[data_type])

        # Apply ops post augmentation.
        data = self.apply_ops(data, self.post_aug_ops)
        data = self.apply_ops(data, self.full_data_post_aug_ops, full_data=True)
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

        data['is_flipped'] = is_flipped
        data['key'] = keys
        data['original_h_w'] = torch.IntTensor([
            self.augmentor.original_h, self.augmentor.original_w])

        # Apply full data ops.
        data = self.apply_ops(data, self.full_data_ops, full_data=True)

        return data

    def __getitem__(self, index):
        return self._getitem(index, concat=True)
