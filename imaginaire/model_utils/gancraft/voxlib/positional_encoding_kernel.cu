// Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, check out LICENSE.md

#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <time.h>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


struct PE_Params {
    int ndegrees;
    int pre_size;
    int post_size;
    bool incl_orig;
};

// const int TILE_DIM_X = 16;  // channel dim
// const int TILE_DIM_Y = 64;  // entry dim
// dim3 dimGrid((p.post_size+TILE_DIM_X-1)/TILE_DIM_X, (p.pre_size+TILE_DIM_Y-1)/TILE_DIM_Y, 1);
// dim3 dimBlock(TILE_DIM_X, TILE_DIM_Y, 1);
template <int TILE_DIM_X, int TILE_DIM_Y, int DUP_Y>
__global__ void positional_encoding_kernel(
    float* __restrict__ out_feature,
    const float* __restrict__ in_feature, const PE_Params p) {

    const int idx_feat = blockIdx.x * TILE_DIM_X + threadIdx.x;
    const int idx_entry_base = blockIdx.y * TILE_DIM_Y * DUP_Y + threadIdx.y * DUP_Y;
    if (idx_feat >= p.post_size) {
        return;
    }

    int stride = p.ndegrees*2;
    if (p.incl_orig) {
        stride += 1;
    }

    for (int j=0; j<DUP_Y; j++) {
        int idx_entry = idx_entry_base + j;
        if (idx_entry >= p.pre_size) {
            return;
        }
        float data = in_feature[idx_entry*p.post_size + idx_feat];

        for (int i=0; i<p.ndegrees; i++) {
            float rad = data * CUDART_PI_F * exp2f(i);
            //float rad = scalbnf(data * CUDART_PI_F, i);
            float sinrad, cosrad;
            sincosf(rad, &sinrad, &cosrad);
            out_feature[idx_entry*p.post_size*stride + i*2*p.post_size + idx_feat] = sinrad;
            out_feature[idx_entry*p.post_size*stride + (i*2+1)*p.post_size + idx_feat] = cosrad;
        }
        if (p.incl_orig) {
            out_feature[idx_entry*p.post_size*stride + (stride-1)*p.post_size + idx_feat] = data;
        }
    }
}

template <int TILE_DIM_X, int TILE_DIM_Y, int DUP_Y>
__global__ void positional_encoding_backward_kernel(
    float* __restrict__ in_feature_grad,
    const float* __restrict__ out_feature_grad, const float* __restrict__ out_feature, const PE_Params p) {

    int idx_feat = blockIdx.x * TILE_DIM_X + threadIdx.x;
    const int idx_entry_base = blockIdx.y * TILE_DIM_Y * DUP_Y + threadIdx.y * DUP_Y;

    if (idx_feat >= p.post_size) {
        return;
    }

    int stride = p.ndegrees*2;
    if (p.incl_orig) {
        stride += 1;
    }

    for (int j=0; j<DUP_Y; j++) {
        int idx_entry = idx_entry_base + j;
        if (idx_entry >= p.pre_size) {
            return;
        }

        float grad = 0.0f;
        for (int i=0; i<p.ndegrees; i++) {
            float grad_t;

            grad_t = out_feature_grad[idx_entry*p.post_size*stride + i*2*p.post_size + idx_feat] *
                out_feature[idx_entry*p.post_size*stride + (i*2+1)*p.post_size + idx_feat];        // cos(x*pi*(2^i))

            grad_t -= out_feature_grad[idx_entry*p.post_size*stride + (i*2+1)*p.post_size + idx_feat] *
                out_feature[idx_entry*p.post_size*stride + (i*2)*p.post_size + idx_feat];        // -sin(x*pi*(2^i))

            grad += grad_t * CUDART_PI_F * exp2f(i);
        }
        if (p.incl_orig) {
            grad += out_feature_grad[idx_entry*p.post_size*stride + (stride-1)*p.post_size + idx_feat];
        }

        in_feature_grad[idx_entry*p.post_size + idx_feat] = grad;
    }
}


// Input:
//      in_feature:     float32 [..., N, ...]
//      ndegree:        int32   Degrees of PE encoding
//      dim:            int32   Dimension to concatenate
//      incl_orig:      bool    Whether to include original feature vector or not
// Output:
//      out_feature:     float32 [..., N*ndegree*2+incl_orig, ...]
// std::vector<torch::Tensor>
torch::Tensor positional_encoding_cuda(const torch::Tensor& in_feature, int ndegrees, int dim, bool incl_orig) {
    CHECK_CUDA(in_feature);

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    torch::Device device = in_feature.device();

    assert(in_feature.dtype() == torch::kFloat32);

    // Handle negative index
    if (dim < 0) {
        dim = in_feature.dim() + dim;
    }
    assert(dim >= 0 && dim < in_feature.dim());

    // No need to be contiguous. Input and output has the same memory layout.
    CHECK_CONTIGUOUS(in_feature);

    PE_Params p;
    p.ndegrees = ndegrees;
    p.incl_orig = incl_orig;

    // This only works for contiguous tensors...
    int pre_size = 1;
    int post_size = 1;
    for (int i=0; i<dim; i++) {
        pre_size *= in_feature.size(i);
    }
    for (int i=dim; i<in_feature.dim(); i++) {
        post_size *= in_feature.size(i);
    }
    p.pre_size = pre_size;
    p.post_size = post_size;

    // Calculate output shape
    std::vector<int64_t> out_feature_shape;
    for (int i=0; i<in_feature.dim(); i++) {
        int64_t dim_t = in_feature.size(i);
        if (i == dim) {
            if (incl_orig) {
                dim_t = dim_t*(ndegrees*2+1);
            } else {
                dim_t = dim_t*ndegrees*2;
            }
        }
        out_feature_shape.push_back(dim_t);
    }

    // Always produce contiguous output
    torch::Tensor out_feature = torch::empty(out_feature_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Launch CUDA kernel
    // Case 1: Concat at the last dimension (post_size < pre_size)  -->  Each thread handle a single post_size
    // Case 2: Concat at the middle (post_size > pre_size)  -->  Each thread handle
    const int TILE_DIM_X = 16;  // channel dim
    const int TILE_DIM_Y = 64;  // entry dim
    //const int DUP_Y = 4; // Each thread handle multiple entries to save threads
    const int DUP_Y = 8; // DGXA 64 samples per ray @ 256x256
    dim3 dimGrid((p.post_size+TILE_DIM_X-1)/TILE_DIM_X, (p.pre_size+(TILE_DIM_Y*DUP_Y)-1)/(TILE_DIM_Y*DUP_Y), 1);
    dim3 dimBlock(TILE_DIM_X, TILE_DIM_Y, 1);
    positional_encoding_kernel<TILE_DIM_X, TILE_DIM_Y, DUP_Y><<<dimGrid, dimBlock, 0, stream>>>(
        out_feature.data_ptr<float>(),
        in_feature.data_ptr<float>(), p
    );

    THCudaCheck(cudaGetLastError());
    return out_feature;
}

//in_feature_grad = voxrender_op.positional_encoding_backward(out_feature_grad, out_feature, ctx.pe_degrees, ctx.dim, ctx.incl_orig);
// Input:
//      out_feature_grad:   float32 [..., N*ndegree*2+incl_orig, ...]
//      out_feature:        float32 [..., N*ndegree*2+incl_orig, ...]
//      ndegrees:           int32   Degrees of PE encoding
//      dim:                int32   Dimension to concatenate
//      incl_orig:          bool    Whether to include original feature vector or not
// Output:
//      in_feature_grad:    float32 [..., N, ...]
// std::vector<torch::Tensor>
torch::Tensor positional_encoding_backward_cuda(const torch::Tensor& out_feature_grad_, const torch::Tensor& out_feature, int ndegrees, int dim, bool incl_orig) {
    CHECK_CUDA(out_feature_grad_);
    CHECK_CUDA(out_feature);

    const torch::Tensor out_feature_grad = out_feature_grad_.contiguous();

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    torch::Device device = out_feature_grad.device();

    assert(out_feature_grad.dtype() == torch::kFloat32);
    assert(out_feature.dtype() == torch::kFloat32);
    assert(out_feature_grad.sizes() == out_feature.sizes());

    // Handle negative index
    if (dim < 0) {
        dim = out_feature.dim() + dim;
    }
    assert(dim >= 0 && dim < out_feature.dim());

    CHECK_CONTIGUOUS(out_feature_grad);
    CHECK_CONTIGUOUS(out_feature);

    PE_Params p;
    p.ndegrees = ndegrees;
    p.incl_orig = incl_orig;

    int expansion_factor = ndegrees*2;
    if (incl_orig) {
        expansion_factor += 1;
    }
    // This only works for contiguous tensors...
    int pre_size = 1;
    int post_size = 1;
    for (int i=0; i<dim; i++) {
        pre_size *= out_feature.size(i);
    }
    for (int i=dim; i<out_feature.dim(); i++) {
        post_size *= out_feature.size(i);
    }
    post_size = post_size / expansion_factor;
    p.pre_size = pre_size;
    p.post_size = post_size;

    // Calculate output shape
    std::vector<int64_t> out_feature_shape;
    for (int i=0; i<out_feature.dim(); i++) {
        int64_t dim_t = out_feature.size(i);
        if (i == dim) {
            dim_t = dim_t / expansion_factor;
        }
        out_feature_shape.push_back(dim_t);
    }

    // Always produce contiguous output
    torch::Tensor in_feature_grad = torch::empty(out_feature_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));


    // Launch CUDA kernel
    // Case 1: Concat at the last dimension (post_size < pre_size)  -->  Each thread handle a single post_size
    // Case 2: Concat at the middle (post_size > pre_size)  -->  Each thread handle
    const int TILE_DIM_X = 16;  // channel dim
    const int TILE_DIM_Y = 64;  // entry dim
    //const int DUP_Y = 4; // Nothing to amortize
    const int DUP_Y = 8; // DGXA
    dim3 dimGrid((p.post_size+TILE_DIM_X-1)/TILE_DIM_X, (p.pre_size+(TILE_DIM_Y*DUP_Y)-1)/(TILE_DIM_Y*DUP_Y), 1);
    dim3 dimBlock(TILE_DIM_X, TILE_DIM_Y, 1);
    positional_encoding_backward_kernel<TILE_DIM_X, TILE_DIM_Y, DUP_Y><<<dimGrid, dimBlock, 0, stream>>>(
        in_feature_grad.data_ptr<float>(),
        out_feature_grad.data_ptr<float>(), out_feature.data_ptr<float>(), p
    );

    THCudaCheck(cudaGetLastError());

    return in_feature_grad;
}
