# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# flake8: noqa: E712
"""Utils for handling datasets."""

import numpy as np
from PIL import Image

# https://github.com/albumentations-team/albumentations#comments
import cv2
from imaginaire.utils.distributed import master_only_print as print
import albumentations as alb  # noqa nopep8
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'ppm', 'bmp',
                  'pgm', 'tif', 'tiff', 'webp',
                  'JPG', 'JPEG', 'PNG', 'PPM', 'BMP',
                  'PGM', 'TIF', 'TIFF', 'WEBP')
VIDEO_EXTENSIONS = 'mp4'


class Augmentor(object):
    r"""Handles data augmentation using albumentations library."""

    def __init__(self, aug_list, image_data_types, interpolators,
                 keypoint_data_types):
        r"""Initializes augmentation pipeline.

        Args:
            aug_list (list): List of augmentation operations in sequence.
            image_data_types (list): List of keys in expected inputs.
            interpolators (list): List of PIL.Image interpolators associated
                with input. NEAREST means image is segmentation map,
                BILINEAR means continuous valued image.
            keypoint_data_types (list): List of keys which are keypoints.
        """

        self.aug_list = aug_list
        self.image_data_types = image_data_types
        self.interpolators = interpolators
        self.crop_h, self.crop_w = None, None
        self.resize_h, self.resize_w = None, None
        self.resize_smallest_side = None
        self.max_time_step = 1
        self.keypoint_data_types = keypoint_data_types

        self.augment_ops = self._build_augmentation_ops()
        # Both crop and resize can't be none at the same time.
        if self.crop_h is None and self.resize_smallest_side is None and\
                self.resize_h is None:
            raise ValueError('resize_smallest_side, resize_h_w, '
                             'and crop_h_w cannot all be missing.')
        # If resize_smallest_side is given, resize_h_w should not be give.
        if self.resize_smallest_side is not None:
            assert self.resize_h is None, \
                'Cannot have both `resize_smallest_side` and `resize_h_w` set.'
        if self.resize_smallest_side is None and self.resize_h is None:
            self.resize_h, self.resize_w = self.crop_h, self.crop_w

    def _build_augmentation_ops(self):
        r"""Builds sequence of augmentation ops.

        Returns:
            (list of alb.ops): List of augmentation ops.
        """
        augs = []
        for key, value in self.aug_list.items():
            if key == 'resize_smallest_side':
                self.resize_smallest_side = value
            elif key == 'resize_h_w':
                h, w = value.split(',')
                h, w = int(h), int(w)
                self.resize_h, self.resize_w = h, w
            elif key == 'random_resize_h_w_aspect':
                aspect_start, aspect_end = value.find('('), value.find(')')
                aspect = value[aspect_start+1:aspect_end]
                aspect_min, aspect_max = aspect.split(',')
                h, w = value[:aspect_start].split(',')[:2]
                h, w = int(h), int(w)
                aspect_min, aspect_max = float(aspect_min), float(aspect_max)
                augs.append(alb.RandomResizedCrop(
                    h, w, scale=(1, 1),
                    ratio=(aspect_min, aspect_max), always_apply=True, p=1))
                self.resize_h, self.resize_w = h, w
            elif key == 'rotate':
                augs.append(alb.Rotate(
                    limit=value, always_apply=True, p=1))
            elif key == 'random_rotate_90':
                augs.append(alb.RandomRotate90(always_apply=False, p=0.5))
            elif key == 'random_scale_limit':
                augs.append(alb.RandomScale(scale_limit=(0, value), p=1))
            elif key == 'random_crop_h_w':
                h, w = value.split(',')
                h, w = int(h), int(w)
                self.crop_h, self.crop_w = h, w
                augs.append(alb.RandomCrop(h, w, always_apply=True, p=1))
            elif key == 'center_crop_h_w':
                h, w = value.split(',')
                h, w = int(h), int(w)
                self.crop_h, self.crop_w = h, w
                augs.append(alb.CenterCrop(h, w, always_apply=True, p=1))
            elif key == 'horizontal_flip':
                # This is handled separately as we need to keep track if this
                # was applied in order to correctly modify keypoint data.
                if value:
                    augs.append(alb.HorizontalFlip(always_apply=False, p=0.5))
            elif key == 'max_time_step':
                self.max_time_step = value
                assert self.max_time_step >= 1, \
                    'max_time_step has to be at least 1'
            else:
                raise ValueError('Unknown augmentation %s' % (key))
        return augs

    def _choose_image_key(self, inputs):
        r"""Choose key to replace with 'image' for input to albumentations.

        Returns:
            key (str): Chosen key to be replace with 'image'
        """
        if 'image' in inputs:
            return 'image'
        for data_type in inputs:
            if data_type in self.image_data_types:
                return data_type

    def _choose_keypoint_key(self, inputs):
        r"""Choose key to replace with 'keypoints' for input to albumentations.
        Returns:
            key (str): Chosen key to be replace with 'keypoints'
        """
        if not self.keypoint_data_types:
            return None
        if 'keypoints' in inputs:
            return 'keypoints'
        for data_type in inputs:
            if data_type in self.keypoint_data_types:
                return data_type

    def _create_augmentation_targets(self, inputs):
        r"""Create additional targets as required by the albumentation library.

        Args:
            inputs (dict): Keys are from self.augmentable_data_types. Values can
                be numpy.ndarray or list of numpy.ndarray
                (image or list of images).
        Returns:
            (dict):        
              - targets (dict): Dict containing mapping of keys to image/mask types.
              - new_inputs (dict): Dict containing mapping of keys to data.
        """
        # Get additional target list.
        targets, new_inputs = {}, {}
        for data_type in inputs:
            if data_type in self.keypoint_data_types:
                # Keypoint-type.
                target_type = 'keypoints'
            elif data_type in self.image_data_types:
                # Image-type.
                # Find the target type (image/mask) based on interpolation
                # method.
                interp = self.interpolators[data_type]
                if interp == Image.NEAREST:
                    target_type = 'mask'
                elif interp == Image.BILINEAR:
                    target_type = 'image'
                else:
                    raise NotImplementedError(
                        '%s is not supported yet' % (interp))
            else:
                raise ValueError(
                    'Data type: %s is not image or keypoint' % (data_type))

            current_data_type_inputs = inputs[data_type]
            if not isinstance(current_data_type_inputs, list):
                current_data_type_inputs = [current_data_type_inputs]

            # Create additional_targets and inputs when there are multiples.
            for idx, new_input in enumerate(current_data_type_inputs):
                key = data_type
                if idx > 0:
                    key = '%s::%05d' % (key, idx)
                targets[key] = target_type
                new_inputs[key] = new_input

        return targets, new_inputs

    def _collate_augmented(self, augmented):
        r"""Collate separated images back into sequence, grouped by keys.

        Args:
            augmented (dict): Dict containing frames with keys of the form
            'key', 'key::00001', 'key::00002', ..., 'key::N'.
        Returns:
            (dict):    
              - outputs (dict): Dict with list of collated inputs, i.e. frames of
              - same key are arranged in order ['key', 'key::00001', ..., 'key::N'].
        """
        full_keys = sorted(augmented.keys())
        outputs = {}
        for full_key in full_keys:
            if '::' not in full_key:
                # First occurrence of this key.
                key = full_key
                outputs[key] = []
            else:
                key = full_key.split('::')[0]
            outputs[key].append(augmented[full_key])
        return outputs

    def _get_resize_h_w(self, height, width):
        r"""Get height and width to resize to, given smallest side.

        Args:
            height (int): Input image height.
            width (int): Input image width.
        Returns:
            (dict):   
              - height (int): Height to resize image to.
              - width (int): Width to resize image to.
        """
        if self.resize_smallest_side is None:
            return self.resize_h, self.resize_w

        if height <= width:
            new_height = self.resize_smallest_side
            new_width = int(np.round(new_height * width / float(height)))
        else:
            new_width = self.resize_smallest_side
            new_height = int(np.round(new_width * height / float(width)))
        return new_height, new_width

    def _perform_unpaired_augmentation(self, inputs):
        r"""Perform different data augmentation on different inputs.

        Args:
            inputs (dict): Keys are from self.image_data_types. Values are list
                of numpy.ndarray (list of images).
        
        Returns:
            (dict):
              - augmented (dict): Augmented inputs, with same keys as inputs.
              - is_flipped (dict): Flag which tells if images have been LR flipped.
        """
        # Process each data type separately as this is unpaired augmentation.
        is_flipped = {}
        for data_type in inputs:
            assert data_type in self.image_data_types
            augmented, flipped_flag = self._perform_paired_augmentation(
                {data_type: inputs[data_type]})
            inputs[data_type] = augmented[data_type]
            is_flipped[data_type] = flipped_flag
        return inputs, is_flipped

    def _perform_paired_augmentation(self, inputs):
        r"""Perform same data augmentation on all inputs.

        Args:
            inputs (dict): Keys are from self.augmentable_data_types. Values are
                list of numpy.ndarray (list of images).
        
        Returns:
            (dict):
              - augmented (dict): Augmented inputs, with same keys as inputs.
              - is_flipped (bool): Flag which tells if images have been LR flipped.
        """
        # All input images here have the same size, and the calling function
        # has checked this.
        # Convert PIL images to numpy array.
        for data_type in inputs:
            if data_type in self.keypoint_data_types or \
                    data_type not in self.image_data_types:
                continue
            for idx in range(len(inputs[data_type])):
                value = inputs[data_type][idx]
                # Get resize h, w.
                w, h = value.size
                self.original_h, self.original_w = h, w
                self.resize_h, self.resize_w = self._get_resize_h_w(h, w)
                # Convert to numpy array with 3 dims (H, W, C).
                value = np.array(value)
                if value.ndim == 2:
                    value = value[..., np.newaxis]
                inputs[data_type][idx] = value

        # Add resize op to augmentation ops.
        aug_ops_with_resize = [alb.Resize(
            self.resize_h, self.resize_w, always_apply=1, p=1)] + \
            self.augment_ops

        # Create targets.
        targets, new_inputs = self._create_augmentation_targets(inputs)
        extra_params = {}

        # Albumentation requires a key called 'image' and
        # a key called 'keypoints', if any keypoints are being passed in.
        # Arbitrarily choose one key of image type to be 'image'.
        chosen_image_key = self._choose_image_key(inputs)
        new_inputs['image'] = new_inputs.pop(chosen_image_key)
        targets['image'] = targets.pop(chosen_image_key)
        # Arbitrarily choose one key of keypoint type to be 'keypoints'.
        chosen_keypoint_key = self._choose_keypoint_key(inputs)
        if chosen_keypoint_key is not None:
            new_inputs['keypoints'] = new_inputs.pop(chosen_keypoint_key)
            targets['keypoints'] = targets.pop(chosen_keypoint_key)
            extra_params['keypoint_params'] = alb.KeypointParams(
                format='xy', remove_invisible=False)

        # Do augmentation.
        augmented = alb.ReplayCompose(
            aug_ops_with_resize, additional_targets=targets,
            **extra_params)(**new_inputs)
        augmentation_params = augmented.pop('replay')

        # Check if flipping has occurred.
        is_flipped = False
        for augmentation_param in augmentation_params['transforms']:
            if 'HorizontalFlip' in augmentation_param['__class_fullname__']:
                is_flipped = augmentation_param['applied']
        self.is_flipped = is_flipped

        # Replace the key 'image' with chosen_image_key, same for 'keypoints'.
        augmented[chosen_image_key] = augmented.pop('image')
        if chosen_keypoint_key is not None:
            augmented[chosen_keypoint_key] = augmented.pop('keypoints')

        # Pack images back into a sequence.
        augmented = self._collate_augmented(augmented)

        # Convert keypoint types to np.array from list.
        for data_type in self.keypoint_data_types:
            augmented[data_type] = np.array(augmented[data_type])

        return augmented, is_flipped

    def perform_augmentation(self, inputs, paired):
        r"""Entry point for augmentation.

        Args:
            inputs (dict): Keys are from self.augmentable_data_types. Values are
                list of numpy.ndarray (list of images).
            paired (bool): Apply same augmentation to all input keys?
        """
        # Make sure that all inputs are of same size, else trouble will
        # ensue. This is because different images might have different
        # aspect ratios.
        # Check within data type.
        for data_type in inputs:
            if data_type in self.keypoint_data_types or \
                    data_type not in self.image_data_types:
                continue
            for idx in range(len(inputs[data_type])):
                if idx == 0:
                    w, h = inputs[data_type][idx].size
                else:
                    this_w, this_h = inputs[data_type][idx].size
                    # assert this_w == w and this_h == h
                    # assert this_w / (1.0 * this_h) == w / (1.0 * h)
                    if this_w / (1.0 * this_h) != w / (1.0 * h):
                        print('(%d, %d) != (%d, %d)' % (
                            this_w, this_h, w, h))
        # Check across data types.
        if paired and self.resize_smallest_side is not None:
            for idx, data_type in enumerate(inputs):
                if data_type in self.keypoint_data_types or \
                        data_type not in self.image_data_types:
                    continue
                if idx == 0:
                    w, h = inputs[data_type][0].size
                else:
                    this_w, this_h = inputs[data_type][0].size
                    # assert this_w == w and this_h == h
                    # assert this_w / (1.0 * this_h) == w / (1.0 * h)
                    if this_w / (1.0 * this_h) != w / (1.0 * h):
                        print('(%d, %d) != (%d, %d)' % (
                            this_w, this_h, w, h))
        # Do appropriate augmentation.
        if paired:
            return self._perform_paired_augmentation(inputs)
        else:
            return self._perform_unpaired_augmentation(inputs)


def load_from_lmdb(keys, lmdbs):
    r"""Load keys from lmdb handles.

    Args:
        keys (dict): This has data_type as key, and a list of paths into LMDB as
            values.
        lmdbs (dict): This has data_type as key, and LMDB handle as value.
    Returns:
        data (dict): This has data_type as key, and a list of decoded items from
            LMDBs as value.
    """
    data = {}
    for data_type in keys:
        if data_type not in data:
            data[data_type] = []
        data_type_keys = keys[data_type]
        if not isinstance(data_type_keys, list):
            data_type_keys = [data_type_keys]
        for key in data_type_keys:
            data[data_type].append(lmdbs[data_type].getitem_by_path(
                key.encode(), data_type))
    return data


def load_from_folder(keys, handles):
    r"""Load keys from lmdb handles.

    Args:
        keys (dict): This has data_type as key, and a list of paths as
            values.
        handles (dict): This has data_type as key, and Folder handle as value.
    Returns:
        data (dict): This has data_type as key, and a list of decoded items from
            folders as value.
    """
    data = {}
    for data_type in keys:
        if data_type not in data:
            data[data_type] = []
        data_type_keys = keys[data_type]
        if not isinstance(data_type_keys, list):
            data_type_keys = [data_type_keys]
        for key in data_type_keys:
            data[data_type].append(handles[data_type].getitem_by_path(
                key.encode(), data_type))
    return data


def get_paired_input_image_channel_number(data_cfg):
    r"""Get number of channels for the input image.

    Args:
        data_cfg (obj): Data configuration structure.
    Returns:
        num_channels (int): Number of input image channels.
    """
    num_channels = 0
    for ix, data_type in enumerate(data_cfg.input_types):
        for k in data_type:
            if k in data_cfg.input_image:
                num_channels += data_type[k].num_channels
                print('Concatenate %s for input.' % data_type)
    print('\tNum. of channels in the input image: %d' % num_channels)
    return num_channels


def get_paired_input_label_channel_number(data_cfg, video=False):
    r"""Get number of channels for the input label map.

    Args:
        data_cfg (obj): Data configuration structure.
        video (bool): Whether we are dealing with video data.
    Returns:
        num_channels (int): Number of input label map channels.
    """
    num_labels = 0
    if not hasattr(data_cfg, 'input_labels'):
        return num_labels
    for ix, data_type in enumerate(data_cfg.input_types):
        for k in data_type:
            if k in data_cfg.input_labels:
                num_labels += data_type[k].num_channels
                if getattr(data_type[k], 'use_dont_care', False):
                    print(data_type[k].use_dont_care)
                    num_labels += 1
            print('Concatenate %s for input.' % data_type)

    if video:
        num_time_steps = getattr(data_cfg.train, 'initial_sequence_length',
                                 None)
        num_labels *= num_time_steps
        num_labels += get_paired_input_image_channel_number(data_cfg) * (
            num_time_steps - 1)

    print('\tNum. of channels in the input label: %d' % num_labels)
    return num_labels


def get_class_number(data_cfg):
    r"""Get number of classes for class-conditional GAN model

    Args:
        data_cfg (obj): Data configuration structure.

    Returns:
        (int): Number of classes.
    """
    return data_cfg.num_classes


def get_crop_h_w(augmentation):
    r"""Get height and width of crop.

    Args:
        augmentation (dict): Dict of applied augmentations.

    Returns:
        (dict):
          - crop_h (int): Height of the image crop.
          - crop_w (int): Width of the image crop.          
    """
    print(augmentation.__dict__.keys())
    for k in augmentation.__dict__.keys():
        if 'crop_h_w' in k:
            filed = augmentation[k]
            crop_h, crop_w = filed.split(',')
            crop_h = int(crop_h)
            crop_w = int(crop_w)
            # assert crop_w == crop_h, 'This implementation only ' \
            #                          'supports square-shaped images.'
            print('\tCrop size: (%d, %d)' % (crop_h, crop_w))
            return crop_h, crop_w
    raise AttributeError
