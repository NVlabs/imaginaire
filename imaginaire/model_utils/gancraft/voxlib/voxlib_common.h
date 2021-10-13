// Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, check out LICENSE.md
#ifndef VOXLIB_COMMON_H
#define VOXLIB_COMMON_H

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")

#include <cuda.h>
#include <cuda_runtime.h>
// CUDA vector math functions
__host__ __device__ __forceinline__ int floor_div(int a, int b) {
    int c = a / b;

    if (c * b > a) {
        c--;
    }

    return c;
}

template <typename scalar_t>
__host__ __forceinline__ void cross(scalar_t* r, const scalar_t* a, const scalar_t* b) {
    r[0] = a[1]*b[2] - a[2]*b[1];
    r[1] = a[2]*b[0] - a[0]*b[2];
    r[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ __host__ __forceinline__ float dot(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename scalar_t, int ndim>
__device__ __host__ __forceinline__ void copyarr(scalar_t* r, const scalar_t* a) {
    #pragma unroll
    for (int i=0; i<ndim; i++) {
        r[i] = a[i];
    }
}

// TODO: use rsqrt to speed up
// inplace version
template <typename scalar_t, int ndim>
__device__ __host__ __forceinline__ void normalize(scalar_t* a) {
    scalar_t vec_len=0.0f;
    #pragma unroll
    for (int i=0; i<ndim; i++) {
        vec_len += a[i]*a[i];
    }
    vec_len = sqrtf(vec_len);
    #pragma unroll
    for (int i=0; i<ndim; i++) {
        a[i] /= vec_len;
    }
}

// normalize + copy
template <typename scalar_t, int ndim>
__device__ __host__ __forceinline__ void normalize(scalar_t* r, const scalar_t* a) {
    scalar_t vec_len=0.0f;
    #pragma unroll
    for (int i=0; i<ndim; i++) {
        vec_len += a[i]*a[i];
    }
    vec_len = sqrtf(vec_len);
    #pragma unroll
    for (int i=0; i<ndim; i++) {
        r[i] = a[i] / vec_len;
    }
}

#endif // VOXLIB_COMMON_H