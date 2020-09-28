# flake8: noqa
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-Xcompiler', '-Wall', '-std=c++14'
]


setup(
    name='channelnorm_cuda',
    py_modules=['channelnorm'],
    ext_modules=[
        CUDAExtension('channelnorm_cuda', [
            './src/channelnorm_cuda.cc',
            './src/channelnorm_kernel.cu'
        ], extra_compile_args={'cxx': ['-Wall', '-std=c++14'],
                               'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
