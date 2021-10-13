# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-fopenmp']
nvcc_args = []

setup(
    name='voxrender',
    ext_modules=[
        CUDAExtension('voxlib', [
            'voxlib.cpp',
            'ray_voxel_intersection.cu',
            'sp_trilinear_worldcoord_kernel.cu',
            'positional_encoding_kernel.cu'
        ],
            extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
