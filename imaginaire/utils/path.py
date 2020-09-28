# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""Utils to deal with directories and paths."""

import glob
import os


def get_immediate_subdirectories(input_dir):
    """List dirs immediately under input_dir.

    Args:
        input_dir (str): Directory to list children of.
        Returns:
        (list): List of directory paths relative to input_dir.
    """
    return sorted([name for name in os.listdir(input_dir)
                   if os.path.isdir(os.path.join(input_dir, name))])


def get_recursive_subdirectories(input_dir, ext):
    """List dirs recursively under input_dir.

    Args:
        input_dir (str): Directory to list children of.
        ext (str): Extension of files expected in this directory.
        Returns:
        (list): List of directory paths relative to input_dir.
    """
    lines = glob.glob('%s/**/*.%s' % (input_dir, ext), recursive=True)
    dirpaths = [os.path.dirname(item) for item in lines]
    dirpaths = [os.path.relpath(item, input_dir) for item in dirpaths]
    dirpaths = sorted(list(set(dirpaths)))
    return dirpaths
