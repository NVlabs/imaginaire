# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md

from imaginaire.datasets.paired_videos import Dataset as VideoDataset


class Dataset(VideoDataset):
    r"""Paired image dataset for use in pix2pixHD, SPADE.

    Args:
        cfg (Config): Loaded config object.
        is_inference (bool): In train or inference mode?
    """

    def __init__(self, cfg, is_inference=False, is_test=False):
        super(Dataset, self).__init__(cfg, is_inference,
                                      sequence_length=1,
                                      is_test=is_test)
        self.is_video_dataset = False

    def _create_mapping(self):
        r"""Creates mapping from idx to key in LMDB.

        Returns:
            (tuple):
              - self.mapping (list): List mapping idx to key.
              - self.epoch_length (int): Number of samples in an epoch.
        """
        idx_to_key = []
        for lmdb_idx, sequence_list in enumerate(self.sequence_lists):
            for sequence_name, filenames in sequence_list.items():
                for filename in filenames:
                    idx_to_key.append({
                        'lmdb_root': self.lmdb_roots[lmdb_idx],
                        'lmdb_idx': lmdb_idx,
                        'sequence_name': sequence_name,
                        'filenames': [filename],
                    })
        self.mapping = idx_to_key
        self.epoch_length = len(self.mapping)
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
        assert self.sequence_length == 1, \
            'Image dataset can only have sequence length = 1, not %d' % (
                self.sequence_length)
        return self.mapping[index]

    def set_sequence_length(self, sequence_length):
        r"""Set the length of sequence you want as output from dataloader.
        Ignore this as this is an image loader.

        Args:
            sequence_length (int): Length of output sequences.
        """
        pass

    def set_inference_sequence_idx(self, index):
        r"""Get frames from this sequence during inference.
        Overriden from super as this is not applicable for images.

        Args:
            index (int): Index of inference sequence.
        """
        raise RuntimeError('Image dataset does not have sequences.')

    def num_inference_sequences(self):
        r"""Number of sequences available for inference.
        Overriden from super as this is not applicable for images.

        Returns:
            (int)
        """
        raise RuntimeError('Image dataset does not have sequences.')
