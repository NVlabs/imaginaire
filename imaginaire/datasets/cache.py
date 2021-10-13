# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import diskcache

"""
INFO:
Cache objects are thread-safe and may be shared between threads.
Two Cache objects may also reference the same directory from separate
threads or processes. In this way, they are also process-safe and support
cross-process communication.
"""


class Cache(object):
    r"""This creates an on disk cache, which saves files as bytes.
    Args:
        root (str): Path to the cache dir.
        size_MB (float): Size of cache in MB.
    """

    def __init__(self, root, size_GB):
        self.root = root
        self.size_limit_B = size_GB * 1024 * 1024 * 1024
        self.cache = diskcache.Cache(root, size_limit=self.size_limit_B)
        print('Created cache of max size %d GB at %s' %
              (size_GB, self.cache.directory))

    def read(self, key):
        if key in self.cache:
            return self.cache[key]
        return False

    def write(self, key, value):
        try:
            self.cache[key] = value
        except Exception as e:  # noqa
            print(e)
            return False
