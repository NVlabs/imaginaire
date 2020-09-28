# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import copy
import random

import torch

from imaginaire.datasets.paired_videos import Dataset as VideoDataset
from imaginaire.model_utils.fs_vid2vid import select_object
from imaginaire.utils.distributed import master_only_print as print


class Dataset(VideoDataset):
    r"""Paired video dataset for use in few-shot vid2vid.

    Args:
        cfg (Config): Loaded config object.
        is_inference (bool): In train or inference mode?
        sequence_length (int): What sequence of images to provide?
        few_shot_K (int): How many images to provide for few-shot?
    """

    def __init__(self, cfg, is_inference=False, sequence_length=None,
                 few_shot_K=None, is_test=False):
        # Get initial few shot K.
        if few_shot_K is None:
            self.few_shot_K = cfg.data.initial_few_shot_K
        else:
            self.few_shot_K = few_shot_K
        # Initialize.
        super(Dataset, self).__init__(
            cfg, is_inference, sequence_length=sequence_length, is_test=is_test)

    def set_inference_sequence_idx(self, index, k_shot_index,
                                   k_shot_frame_index):
        r"""Get frames from this sequence during inference.

        Args:
            index (int): Index of inference sequence.
            k_shot_index (int): Index of sequence from which k_shot is sampled.
            k_shot_frame_index (int): Index of frame to sample.
        """
        assert self.is_inference
        assert index < len(self.mapping)
        assert k_shot_index < len(self.mapping)
        assert k_shot_frame_index < len(self.mapping[k_shot_index])

        self.inference_sequence_idx = index
        self.inference_k_shot_sequence_index = k_shot_index
        self.inference_k_shot_frame_index = k_shot_frame_index
        self.epoch_length = len(
            self.mapping[self.inference_sequence_idx]['filenames'])

    def set_sequence_length(self, sequence_length, few_shot_K=None):
        r"""Set the length of sequence you want as output from dataloader.

        Args:
            sequence_length (int): Length of output sequences.
            few_shot_K (int): Number of few-shot frames.
        """
        if few_shot_K is None:
            few_shot_K = self.few_shot_K
        assert isinstance(sequence_length, int)
        assert isinstance(few_shot_K, int)
        if (sequence_length + few_shot_K) > self.sequence_length_max:
            error_message = \
                'Requested sequence length (%d) ' % (sequence_length) + \
                '+ few shot K (%d) > ' % (few_shot_K) + \
                'max sequence length (%d). ' % (self.sequence_length_max)
            print(error_message)
            sequence_length = self.sequence_length_max - few_shot_K
            print('Reduced sequence length to %s' % (sequence_length))
        self.sequence_length = sequence_length
        self.few_shot_K = few_shot_K
        # Recalculate mapping as some sequences might no longer be useful.
        self.mapping, self.epoch_length = self._create_mapping()
        print('Epoch length:', self.epoch_length)

    def _create_mapping(self):
        r"""Creates mapping from idx to key in LMDB.

        Returns:
            (tuple):
              - self.mapping (dict): Dict of seq_len to list of sequences.
              - self.epoch_length (int): Number of samples in an epoch.
        """
        # Create dict mapping length to sequence.
        length_to_key, num_selected_seq = {}, 0
        has_additional_lists = len(self.additional_lists) > 0
        for lmdb_idx, sequence_list in enumerate(self.sequence_lists):
            for sequence_name, filenames in sequence_list.items():
                if len(filenames) >= (self.sequence_length + self.few_shot_K):
                    if len(filenames) not in length_to_key:
                        length_to_key[len(filenames)] = []
                    if has_additional_lists:
                        obj_indices = self.additional_lists[lmdb_idx][
                            sequence_name]
                    else:
                        obj_indices = [0 for _ in range(len(filenames))]
                    length_to_key[len(filenames)].append({
                        'lmdb_root': self.lmdb_roots[lmdb_idx],
                        'lmdb_idx': lmdb_idx,
                        'sequence_name': sequence_name,
                        'filenames': filenames,
                        'obj_indices': obj_indices,
                    })
                    num_selected_seq += 1
        self.mapping = length_to_key
        self.epoch_length = num_selected_seq

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
            chosen_obj_indices = [chosen_sequence['obj_indices'][index]]
            k_shot_chosen_sequence = self.mapping[
                self.inference_k_shot_sequence_index]
            k_shot_chosen_filenames = [k_shot_chosen_sequence['filenames'][
                                       self.inference_k_shot_frame_index]]
            k_shot_chosen_obj_indices = [k_shot_chosen_sequence['obj_indices'][
                                         self.inference_k_shot_frame_index]]
            # Prepare few shot key.
            few_shot_key = copy.deepcopy(k_shot_chosen_sequence)
            few_shot_key['filenames'] = k_shot_chosen_filenames
            few_shot_key['obj_indices'] = k_shot_chosen_obj_indices
        else:
            # Pick a time step for temporal augmentation.
            time_step = random.randint(1, self.augmentor.max_time_step)
            required_sequence_length = 1 + \
                (self.sequence_length - 1) * time_step

            # If step is too large, default to step size of 1.
            if required_sequence_length + self.few_shot_K > \
                    self.sequence_length_max:
                required_sequence_length = self.sequence_length
                time_step = 1

            # Find valid sequences.
            valid_sequences = []
            for sequence_length, sequences in self.mapping.items():
                if sequence_length >= required_sequence_length + \
                        self.few_shot_K:
                    valid_sequences.extend(sequences)

            # Pick a sequence.
            chosen_sequence = random.choice(valid_sequences)

            # Choose filenames.
            max_start_idx = len(chosen_sequence['filenames']) - \
                required_sequence_length
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + required_sequence_length
            chosen_filenames = chosen_sequence['filenames'][
                start_idx:end_idx:time_step]
            chosen_obj_indices = chosen_sequence['obj_indices'][
                start_idx:end_idx:time_step]

            # Find the K few shot filenames.
            valid_range = list(range(start_idx)) + \
                list(range(end_idx, len(chosen_sequence['filenames'])))
            k_shot_chosen = sorted(random.sample(valid_range, self.few_shot_K))
            k_shot_chosen_filenames = [chosen_sequence['filenames'][idx]
                                       for idx in k_shot_chosen]
            k_shot_chosen_obj_indices = [chosen_sequence['obj_indices'][idx]
                                         for idx in k_shot_chosen]
            assert not (set(chosen_filenames) & set(k_shot_chosen_filenames))

            assert len(chosen_filenames) == self.sequence_length
            assert len(k_shot_chosen_filenames) == self.few_shot_K

            # Prepare few shot key.
            few_shot_key = copy.deepcopy(chosen_sequence)
            few_shot_key['filenames'] = k_shot_chosen_filenames
            few_shot_key['obj_indices'] = k_shot_chosen_obj_indices

        # Prepre output key.
        key = copy.deepcopy(chosen_sequence)
        key['filenames'] = chosen_filenames
        key['obj_indices'] = chosen_obj_indices
        return key, few_shot_key

    def _prepare_data(self, keys, concat):
        r"""Load data and perform augmentation.

        Args:
            keys (dict): Key into LMDB/folder dataset for this item.
            concat (bool): Concatenate all items in labels?
        Returns:
            data (dict): Dict with all chosen data_types.
        """
        # Unpack keys.
        lmdb_idx = keys['lmdb_idx']
        sequence_name = keys['sequence_name']
        filenames = keys['filenames']
        obj_indices = keys['obj_indices']

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

        # Select the object in data using the object indices.
        data = select_object(data, obj_indices)

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

        data['is_flipped'] = is_flipped
        data['key'] = keys

        # Add keypoint xy to data.
        data.update(kp_data)

        return data

    def _getitem(self, index, concat=True):
        r"""Gets selected files.

        Args:
            index (int): Index into dataset.
            concat (bool): Concatenate all items in labels?
        Returns:
            data (dict): Dict with all chosen data_types.
        """
        # Select a sample from the available data.
        keys, few_shot_keys = self._sample_keys(index)

        data = self._prepare_data(keys, concat)
        few_shot_data = self._prepare_data(few_shot_keys, concat)

        # Add few shot data into data.
        for key, value in few_shot_data.items():
            data['few_shot_' + key] = few_shot_data[key]

        # Apply full data ops.
        if self.is_inference:
            if index == 0:
                pass
            elif index < self.cfg.data.num_workers:
                data_0 = self._getitem(0)
                if 'common_attr' in data_0:
                    self.common_attr = data['common_attr'] = \
                        data_0['common_attr']
            else:
                if hasattr(self, 'common_attr'):
                    data['common_attr'] = self.common_attr

        data = self.apply_ops(data, self.full_data_ops, full_data=True)

        if self.is_inference and index == 0 and 'common_attr' in data:
            self.common_attr = data['common_attr']

        return data
