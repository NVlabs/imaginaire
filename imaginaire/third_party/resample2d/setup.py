# flake8: noqa
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


cuda_version = os.getenv('CUDA_VERSION')
print('CUDA_VERSION: {}'.format(cuda_version))

nvcc_args = list()
# nvcc_args.append('-gencode')
# nvcc_args.append('arch=compute_50,code=sm_50')
# nvcc_args.append('-gencode')
# nvcc_args.append('arch=compute_52,code=sm_52')
# nvcc_args.append('-gencode')
# nvcc_args.append('arch=compute_60,code=sm_60')
# nvcc_args.append('-gencode')
# nvcc_args.append('arch=compute_61,code=sm_61')
nvcc_args.append('-gencode')
nvcc_args.append('arch=compute_70,code=sm_70')
nvcc_args.append('-gencode')
nvcc_args.append('arch=compute_75,code=sm_75')
if cuda_version is not None:
    if cuda_version >= '11.0':
        nvcc_args.append('-gencode')
        nvcc_args.append('arch=compute_80,code=sm_80')
nvcc_args.append('-Xcompiler')
nvcc_args.append('-Wall')
nvcc_args.append('-std=c++14')

setup(
    name='resample2d_cuda',
    py_modules=['resample2d'],
    ext_modules=[
        CUDAExtension('resample2d_cuda', [
            './src/resample2d_cuda.cc',
            './src/resample2d_kernel.cu'
        ], extra_compile_args={'cxx': ['-Wall', '-std=c++14'],
                               'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
