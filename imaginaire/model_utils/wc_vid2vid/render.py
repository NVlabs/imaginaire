# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import pickle
import time

import numpy as np


class SplatRenderer(object):
    """Splatting 3D point cloud into image using precomputed mapping."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the renderer."""
        # 1 = point seen before, 0 = not seen.
        # This is numpy uint8 array of size (N, 1)
        self.seen_mask = None

        # Time of first colorization of 3D point.
        # This is numpy uint16 array of size (N, 1)
        self.seen_time = None

        # colors[kp_idx] is color of kp_idx'th keypoint.
        # This is a numpy uint8 array of size (N, 3)
        self.colors = None

        self.time_taken = 0
        self.call_idx = 0

    def num_points(self):
        r"""Number of points with assigned colors."""
        return np.sum(self.seen_mask)

    def _resize_arrays(self, max_point_idx):
        r"""Makes arrays bigger, if needed.
        Args:
            max_point_idx (int): Highest 3D point index seen so far.
        """
        if self.colors is None:
            old_max_point_idx = 0
        else:
            old_max_point_idx = self.colors.shape[0]

        if max_point_idx > old_max_point_idx:
            # Init new bigger arrays.
            colors = np.zeros((max_point_idx, 3), dtype=np.uint8)
            seen_mask = np.zeros((max_point_idx, 1), dtype=np.uint8)
            seen_time = np.zeros((max_point_idx, 1), dtype=np.uint16)
            # Copy old colors, if exist.
            if old_max_point_idx > 0:
                colors[:old_max_point_idx] = self.colors
                seen_mask[:old_max_point_idx] = self.seen_mask
                seen_time[:old_max_point_idx] = self.seen_time
            # Reset pointers.
            self.colors = colors
            self.seen_mask = seen_mask
            self.seen_time = seen_time

    def update_point_cloud(self, image, point_info):
        r"""Updates point cloud with new points and colors.
        Args:
            image (H x W x 3, uint8): Select colors from this image to assign to
            3D points which do not have previously assigned colors.
            point_info (N x 3): (i, j, 3D point idx) per row containing
            mapping of image pixel to 3D point in point cloud.
        """
        if point_info is None or len(point_info) == 0:
            return

        start = time.time()
        self.call_idx += 1

        i_idxs = point_info[:, 0]
        j_idxs = point_info[:, 1]
        point_idxs = point_info[:, 2]

        # Allocate memory for new colors.
        max_point_idx = np.max(np.array(point_idxs)) + 1
        self._resize_arrays(max_point_idx)
        # print('max point idx:', max_point_idx)

        # Save only the new colors.
        self.colors[point_idxs] = \
            self.seen_mask[point_idxs] * self.colors[point_idxs] + \
            (1 - self.seen_mask[point_idxs]) * image[i_idxs, j_idxs]

        # Save point seen times.
        self.seen_time[point_idxs] = \
            self.seen_mask[point_idxs] * self.seen_time[point_idxs] + \
            (1 - self.seen_mask[point_idxs]) * self.call_idx

        # Update seen point mask.
        self.seen_mask[point_idxs] = 1

        end = time.time()
        self.time_taken += (end - start)

    def render_image(self, point_info, w, h, return_mask=False):
        r"""Creates image of (h, w) and fills in colors.
        Args:
            point_info (N x 3): (i, j, 3D point idx) per row containing
            mapping of image pixel to 3D point in point cloud.
            w (int): Width of output image.
            h (int): Height of output image.
            return_mask (bool): Return binary mask of coloring.
        Returns:
            (tuple):
              - output (H x W x 3, uint8): Image formed with mapping and colors.
              - mask (H x W x 1, uint8): Binary (255 or 0) mask of colorization.
        """
        output = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w, 1), dtype=np.uint8)

        if point_info is None or len(point_info) == 0:
            if return_mask:
                return output, mask
            else:
                return output

        start = time.time()

        i_idxs = point_info[:, 0]
        j_idxs = point_info[:, 1]
        point_idxs = point_info[:, 2]

        # Allocate memory for new colors.
        max_point_idx = np.max(np.array(point_idxs)) + 1
        self._resize_arrays(max_point_idx)

        # num_found = np.sum(self.seen_mask[point_idxs])
        # print('Found %d points to color' % (num_found))

        # Copy colors.
        output[i_idxs, j_idxs] = self.colors[point_idxs]

        end = time.time()
        self.time_taken += (end - start)

        if return_mask:
            mask[i_idxs, j_idxs] = 255 * self.seen_mask[point_idxs]
            return output, mask
        else:
            return output


def decode_unprojections(data):
    r"""Unpickle unprojections and make array.
    Args:
        data (array of pickled info): Each pickled string has keypoint mapping
        info.
    Returns:
        output (dict): Keys are the different resolutions, and values are padded
        mapping information.
    """

    # Unpickle unprojections and store them in a dict with resolutions as keys.
    all_unprojections = {}
    for item in data:
        info = pickle.loads(item)

        for resolution, value in info.items():
            if resolution not in all_unprojections:
                all_unprojections[resolution] = []

            if not value or value is None:
                point_info = []
            else:
                point_info = value
            all_unprojections[resolution].append(point_info)

    outputs = {}
    for resolution, values in all_unprojections.items():
        # Get max length of mapping.
        max_len = 0
        for value in values:
            max_len = max(max_len, len(value))
            # Entries are a 3-tuple of (i_idx, j_idx, point_idx).
            assert len(value) % 3 == 0

        # Pad each mapping to max_len.
        values = [
            value +  # Original info.
            [-1] * (max_len - len(value)) +  # Padding.
            [len(value) // 3] * 3  # End sentinel with length.
            for value in values
        ]

        # Convert each mapping to numpy and reshape.
        values = [np.array(value).reshape(-1, 3) for value in values]

        # Stack and put in output.
        # Shape is (T, N, 3). T is time steps, N is num mappings.
        outputs[resolution] = np.stack(values, axis=0)

    return outputs
