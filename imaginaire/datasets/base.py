# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""All datasets are inherited from this class."""

import importlib
import json
import os
import pickle
from collections import OrderedDict
from functools import partial
from inspect import signature

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from imaginaire.datasets.folder import FolderDataset
from imaginaire.datasets.lmdb import IMG_EXTENSIONS, LMDBDataset
from imaginaire.utils.data import \
    (VIDEO_EXTENSIONS, Augmentor, load_from_folder, load_from_lmdb)
from imaginaire.utils.lmdb import create_metadata


class BaseDataset(data.Dataset):
    r"""Base class for image/video datasets.

    Args:
        cfg (Config object): Input config.
        is_inference (bool): Training if False, else validation.
        is_test (bool): Final test set after training and validation.
    """

    def __init__(self, cfg, is_inference, is_test):
        super(BaseDataset, self).__init__()

        self.cfg = cfg
        self.is_inference = is_inference
        self.is_test = is_test
        if self.is_test:
            self.cfgdata = self.cfg.test_data
            data_info = self.cfgdata.test
        else:
            self.cfgdata = self.cfg.data
            if self.is_inference:
                data_info = self.cfgdata.val
            else:
                data_info = self.cfgdata.train
        self.name = self.cfgdata.name
        self.lmdb_roots = data_info.roots
        self.dataset_is_lmdb = getattr(data_info, 'is_lmdb', True)
        if not self.dataset_is_lmdb:
            assert hasattr(self.cfgdata, 'paired')

        if self.dataset_is_lmdb:
            # Add handle to function to load data from LMDB.
            self.load_from_dataset = load_from_lmdb
        else:
            # Add handle to function to load data from folder.
            self.load_from_dataset = load_from_folder
            # Create metadata for folders.
            print('Creating metadata')
            all_filenames, all_metadata = [], []
            if self.is_test:
                cfg.data_backup = cfg.data
                cfg.data = cfg.test_data
            for root in self.lmdb_roots:
                filenames, metadata = create_metadata(
                    data_root=root, cfg=cfg, paired=self.cfgdata['paired'])
                all_filenames.append(filenames)
                all_metadata.append(metadata)
            if self.is_test:
                cfg.data = cfg.data_backup

        # Get the types of data stored in dataset, and their extensions.
        self.data_types = []  # Names of data types.
        self.dataset_data_types = []  # These data types are in the dataset.
        self.image_data_types = []  # These types are images.
        self.normalize = {}  # Does this data type need normalization?
        self.extensions = {}  # What is this data type's file extension.
        self.interpolators = {}  # Which interpolator to use?
        self.num_channels = {}  # How many channels does this data type have?
        self.pre_aug_ops = {}  # Ops on data type before augmentation.
        self.post_aug_ops = {}  # Ops on data type after augmentation.
        self.use_dont_care = {}  # Use dont care label for this label?

        # Extract info from data types.
        for data_type in self.cfgdata.input_types:
            name = list(data_type.keys())
            assert len(name) == 1
            name = name[0]
            info = data_type[name]

            if 'ext' not in info:
                info['ext'] = None
            if 'normalize' not in info:
                info['normalize'] = False
            if 'interpolator' not in info:
                info['interpolator'] = None
            if 'pre_aug_ops' not in info:
                info['pre_aug_ops'] = 'None'
            if 'post_aug_ops' not in info:
                info['post_aug_ops'] = 'None'
            if 'use_dont_care' not in info:
                info['use_dont_care'] = False
            if 'computed_on_the_fly' not in info:
                info['computed_on_the_fly'] = False
            if 'num_channels' not in info:
                info['num_channels'] = None

            self.data_types.append(name)
            if not info['computed_on_the_fly']:
                self.dataset_data_types.append(name)

            self.extensions[name] = info['ext']
            self.normalize[name] = info['normalize']
            self.num_channels[name] = info['num_channels']
            self.pre_aug_ops[name] = [op.strip() for op in
                                      info['pre_aug_ops'].split(',')]
            self.post_aug_ops[name] = [op.strip() for op in
                                       info['post_aug_ops'].split(',')]
            self.use_dont_care[name] = info['use_dont_care']
            self.interpolators[name] = None
            if info['ext'] is not None and \
                    (info['ext'] in IMG_EXTENSIONS or
                        info['ext'] in VIDEO_EXTENSIONS):
                self.image_data_types.append(name)
                self.interpolators[name] = getattr(
                    Image, info['interpolator'])

        # Add some info into cfgdata for legacy support.
        self.cfgdata.data_types = self.data_types
        self.cfgdata.use_dont_care = [self.use_dont_care[name]
                                      for name in self.data_types]
        self.cfgdata.num_channels = [self.num_channels[name]
                                     for name in self.data_types]

        # Augmentations which need full dict.
        self.full_data_post_aug_ops, self.full_data_ops = [], []
        if hasattr(self.cfgdata, 'full_data_ops'):
            ops = self.cfgdata.full_data_ops
            self.full_data_ops.extend([op.strip() for op in ops.split(',')])
        if hasattr(self.cfgdata, 'full_data_post_aug_ops'):
            ops = self.cfgdata.full_data_post_aug_ops
            self.full_data_post_aug_ops.extend(
                [op.strip() for op in ops.split(',')])

        # These are the labels which will be concatenated for generator input.
        self.input_labels = []
        if hasattr(self.cfgdata, 'input_labels'):
            self.input_labels = self.cfgdata.input_labels

        # These are the keypoints which also need to be augmented.
        self.keypoint_data_types = []
        if hasattr(self.cfgdata, 'keypoint_data_types'):
            self.keypoint_data_types = self.cfgdata.keypoint_data_types

        # Create augmentation operations.
        if is_test:
            aug_list = self.cfgdata.test.augmentations
        else:
            if is_inference:
                aug_list = self.cfgdata.val.augmentations
            else:
                aug_list = self.cfgdata.train.augmentations
        self.augmentor = Augmentor(
            aug_list, self.image_data_types, self.interpolators,
            self.keypoint_data_types)
        self.augmentable_types = self.image_data_types + \
            self.keypoint_data_types

        # Create torch transformations.
        self.transform = {}
        for data_type in self.image_data_types:
            normalize = self.normalize[data_type]
            self.transform[data_type] = self._get_transform(normalize)

        # Initialize handles.
        self.sequence_lists = []  # List of sequences per dataset root.
        self.lmdbs = {}  # Dict for list of lmdb handles per data type.
        for data_type in self.dataset_data_types:
            self.lmdbs[data_type] = []
        self.dataset_probability = None
        self.additional_lists = []

        # Load each dataset.
        for idx, root in enumerate(self.lmdb_roots):
            if self.dataset_is_lmdb:
                self._add_dataset(root)
            else:
                self._add_dataset(root, filenames=all_filenames[idx],
                                  metadata=all_metadata[idx])

        # Compute dataset statistics and create whatever self.variables required
        # for the specific dataloader.
        self._compute_dataset_stats()

        # Build index of data to sample.
        self.mapping, self.epoch_length = self._create_mapping()

    def _create_mapping(self):
        r"""Creates mapping from data sample idx to actual LMDB keys.
            All children need to implement their own.

        Returns:
            self.mapping (list): List of LMDB keys.
        """
        raise NotImplementedError

    def _compute_dataset_stats(self):
        r"""Computes required statistics about dataset.
           All children need to implement their own.
        """
        pass

    def __getitem__(self, index):
        r"""Entry function for dataset."""
        raise NotImplementedError

    def _get_transform(self, normalize):
        r"""Convert numpy to torch tensor.

        Args:
            normalize (bool): Normalize image i.e. (x - 0.5) * 2.
                Goes from [0, 1] -> [-1, 1].
        Returns:
            Composed list of torch transforms.
        """
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))
        return transforms.Compose(transform_list)

    def _add_dataset(self, root, filenames=None, metadata=None):
        r"""Adds an LMDB dataset to a list of datasets.

        Args:
            root (str): Path to LMDB or folder dataset.
            filenames: List of filenames for folder dataset.
            metadata: Metadata for folder dataset.
        """
        # Get sequences associated with this dataset.
        if filenames is None:
            list_path = 'all_filenames.json'
            with open(os.path.join(root, list_path)) as fin:
                sequence_list = OrderedDict(json.load(fin))
        else:
            sequence_list = filenames
        self.sequence_lists.append(sequence_list)

        additional_path = 'all_indices.json'
        if os.path.exists(os.path.join(root, additional_path)):
            print('Using additional list for object indices.')
            with open(os.path.join(root, additional_path)) as fin:
                additional_list = OrderedDict(json.load(fin))
            self.additional_lists.append(additional_list)

        # Get LMDB dataset handles.
        for data_type in self.dataset_data_types:
            if self.dataset_is_lmdb:
                self.lmdbs[data_type].append(
                    LMDBDataset(os.path.join(root, data_type)))
            else:
                self.lmdbs[data_type].append(
                    FolderDataset(os.path.join(root, data_type), metadata))

    def _encode_onehot(self, label_map, num_labels, use_dont_care):
        r"""Make input one-hot.

        Args:
            label_map (torch.Tensor): (C, H, W) tensor containing indices.
            num_labels (int): Number of labels to expand tensor to.
            use_dont_care (bool): Use the dont care label or not?
        Returns:
            output (torch.Tensor): (num_labels, H, W) one-hot tensor.
        """
        # All labels lie in [0. num_label - 1].
        # Encode dont care as num_label.
        label_map[label_map < 0] = num_labels
        label_map[label_map >= num_labels] = num_labels

        size = label_map.size()
        output_size = (num_labels + 1, size[1], size[2])
        output = torch.zeros(*output_size)
        output = output.scatter_(0, label_map.data.long(), 1.0)

        if not use_dont_care:
            # Size of output is (num_labels + 1, H, W).
            # Last label is for dont care segmentation index.
            # Only select first num_labels channels.
            output = output[:num_labels, ...]

        return output

    def perform_augmentation(self, data, paired):
        r"""Perform data augmentation on images only.

        Args:
            data (dict): Keys are from data types. Values can be numpy.ndarray
                or list of numpy.ndarray (image or list of images).
            paired (bool): Apply same augmentation to all input keys?
        Returns:
            (tuple):
              - data (dict): Augmented data, with same keys as input data.
              - is_flipped (bool): Flag which tells if images have been
                left-right flipped.
        """
        aug_inputs = {}
        for data_type in self.augmentable_types:
            aug_inputs[data_type] = data[data_type]

        augmented, is_flipped = self.augmentor.perform_augmentation(
            aug_inputs, paired=paired)

        for data_type in self.augmentable_types:
            data[data_type] = augmented[data_type]

        return data, is_flipped

    def to_tensor(self, data):
        r"""Convert all images to tensor.

        Args:
            data (dict): Dict containing data_type as key, with each value
                as a list of numpy.ndarrays.
        Returns:
            data (dict): Dict containing data_type as key, with each value
            as a list of torch.Tensors.
        """
        for data_type in self.image_data_types:
            for idx in range(len(data[data_type])):
                if data[data_type][idx].dtype == np.uint16:
                    data[data_type][idx] = data[data_type][idx].astype(
                        np.float32)
                data[data_type][idx] = self.transform[data_type](
                    data[data_type][idx])
        return data

    def make_one_hot(self, data):
        r"""Convert appropriate image data types to one-hot representation.

        Args:
            data (dict): Dict containing data_type as key, with each value
                as a list of torch.Tensors.
        Returns:
            data (dict): same as input data, but with one-hot for selected
            types.
        """
        for data_type in self.image_data_types:
            expected_num_channels = self.num_channels[data_type]
            num_channels = data[data_type][0].size(0)
            interpolation_method = self.interpolators[data_type]
            if num_channels < expected_num_channels:
                if num_channels != 1:
                    raise ValueError(
                        'Num channels: %d. ' % (num_channels) +
                        'One-hot expansion can only be done if ' +
                        'image has 1 channel')
                assert interpolation_method == Image.NEAREST, \
                    'Cant do one-hot on image which has been resized' + \
                    'with BILINEAR.'

                if data_type in self.use_dont_care:
                    use_dont_care = self.use_dont_care[data_type]
                else:
                    use_dont_care = False

                for idx in range(len(data[data_type])):
                    data[data_type][idx] = self._encode_onehot(
                        data[data_type][idx] * 255.0, expected_num_channels,
                        use_dont_care)
            elif num_channels > expected_num_channels:
                raise ValueError(
                    'Data type: ' + data_type + ', ' +
                    'Num channels %d > Expected num channels %d' % (
                        num_channels, expected_num_channels))
        return data

    def apply_ops(self, data, op_dict, full_data=False):
        r"""Apply any ops from op_dict to data types.

        Args:
            data (dict): Dict containing data_type as key, with each value
                as a list of numpy.ndarrays.
            op_dict (dict): Dict containing data_type as key, with each value
                containing string of operations to apply.
            full_data (bool): Do these ops require access to the full data?
        Returns:
            data (dict): Dict containing data_type as key, with each value
            modified by the op if any.
        """
        if full_data:
            # op needs entire data dict.
            for op in op_dict:
                if op == 'None':
                    continue
                op, op_type = self.get_op(op)
                assert op_type == 'full_data'
                data = op(data)
        else:
            # op per data type.
            if not op_dict:
                return data
            for data_type in data:
                for op in op_dict[data_type]:
                    if op == 'None':
                        continue
                    op, op_type = self.get_op(op)
                    data[data_type] = op(data[data_type])

                    if op_type == 'vis':
                        # We have converted this data type to an image. Enter it
                        # in self.image_data_types and give it a torch
                        # transform.
                        if data_type not in self.image_data_types:
                            self.image_data_types.append(data_type)
                            normalize = self.normalize[data_type]
                            self.transform[data_type] = \
                                self._get_transform(normalize)
                    elif op_type == 'convert':
                        continue
                    elif op_type is None:
                        continue
                    else:
                        raise NotImplementedError
        return data

    def get_op(self, op):
        r"""Get function to apply for specific op.

        Args:
            op (str): Name of the op.
        Returns:
            function handle.
        """
        def list_to_tensor(data):
            r"""Convert list of numeric values to tensor."""
            assert isinstance(data, list)
            return torch.from_numpy(np.array(data, dtype=np.float32))

        def decode_json_list(data):
            r"""Decode list of strings in json to objects."""
            assert isinstance(data, list)
            return [json.loads(item) for item in data]

        def decode_pkl_list(data):
            r"""Decode list of pickled strings to objects."""
            assert isinstance(data, list)
            return [pickle.loads(item) for item in data]

        def list_to_numpy(data):
            r"""Convert list of numeric values to numpy array."""
            assert isinstance(data, list)
            return np.array(data)

        if op == 'to_tensor':
            return list_to_tensor, None
        elif op == 'decode_json':
            return decode_json_list, None
        elif op == 'decode_pkl':
            return decode_pkl_list, None
        elif op == 'to_numpy':
            return list_to_numpy, None
        elif '::' in op:
            parts = op.split('::')
            if len(parts) == 2:
                module, function = parts
                module = importlib.import_module(module)
                function = getattr(module, function)
                sig = signature(function)
                assert len(sig.parameters) == 3, \
                    'Full data functions take in (cfgdata, full_data, ' \
                    'is_inference) as input.'
                function = partial(function, self.cfgdata, self.is_inference)
                function_type = 'full_data'
            elif len(parts) == 3:
                function_type, module, function = parts
                module = importlib.import_module(module)
                function = getattr(module, function)
                sig = signature(function)
                if function_type == 'vis':
                    if len(sig.parameters) != 9:
                        raise ValueError(
                            'vis function type needs to take ' +
                            '(resize_h, resize_w, crop_h, crop_w, ' +
                            'original_h, original_w, is_flipped, cfgdata, ' +
                            'data) as input.')
                    function = partial(function,
                                       self.augmentor.resize_h,
                                       self.augmentor.resize_w,
                                       self.augmentor.crop_h,
                                       self.augmentor.crop_w,
                                       self.augmentor.original_h,
                                       self.augmentor.original_w,
                                       self.augmentor.is_flipped,
                                       self.cfgdata)
                elif function_type == 'convert':
                    if len(sig.parameters) != 1:
                        raise ValueError(
                            'convert function type needs to take ' +
                            '(data) as input.')
                else:
                    raise ValueError('Unknown op: %s' % (op))
            else:
                raise ValueError('Unknown op: %s' % (op))
            return function, function_type
        else:
            raise ValueError('Unknown op: %s' % (op))

    def __len__(self):
        return self.epoch_length
