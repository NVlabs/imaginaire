# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import math
import os
import pynvml

pynvml.nvmlInit()


def systemGetDriverVersion():
    r"""Get Driver Version"""
    return pynvml.nvmlSystemGetDriverVersion()


def deviceGetCount():
    r"""Get number of devices"""
    return pynvml.nvmlDeviceGetCount()


class device(object):
    r"""Device used for nvml."""
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)

    def __init__(self, device_idx):
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def getName(self):
        r"""Get obect name"""
        return pynvml.nvmlDeviceGetName(self.handle)

    def getCpuAffinity(self):
        r"""Get CPU affinity"""
        affinity_string = ''
        for j in pynvml.nvmlDeviceGetCpuAffinity(
                self.handle, device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = '{:064b}'.format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        return [i for i, e in enumerate(affinity_list) if e != 0]


def set_affinity(gpu_id=None):
    r"""Set GPU affinity

    Args:
        gpu_id (int): Which gpu device.
    """
    if gpu_id is None:
        gpu_id = int(os.getenv('LOCAL_RANK', 0))

    dev = device(gpu_id)
    os.sched_setaffinity(0, dev.getCpuAffinity())

    # list of ints
    # representing the logical cores this process is now affinitied with
    return os.sched_getaffinity(0)
