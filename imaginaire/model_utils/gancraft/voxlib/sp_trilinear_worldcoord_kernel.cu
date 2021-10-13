// Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, check out LICENSE.md
//
// Fast routine for sparse tri-linear interpolation of high dimensional features.
// Ignore label is supported.


#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


struct SpTrilinear_wc_Params {
    int in_feature_dim;
    int in_feature_numentries;
    int corner_lut_dims[3];
    int corner_lut_strides[3];
    int in_worldcoord_dims[8];
    int in_worldcoord_strides[8];
    int in_worldcoord_ndim;
    int out_feature_dims[8];
    int out_feature_strides[8];
    bool ign_zero;
};


// out_feature.data_ptr<float>(),
// in_feature.data_ptr<float>(), corner_lut_t.data_ptr<int32_t>(), in_worldcoord.data_ptr<float>(), p
template <int TILE_DIM_X, int TILE_DIM_Y, int DUP_X>
__global__ void sp_trilinear_worldcoord_kernel(
    float* __restrict__ out_feature,
    const float* __restrict__ in_feature, const int32_t* __restrict__ corner_lut_t, const float* __restrict__ in_worldcoord, SpTrilinear_wc_Params p) {

    const int GRID_X = gridDim.y;
    int idx_entry = blockIdx.x * TILE_DIM_Y + threadIdx.y;

    // Index processing
    //int index[7];
    int t = idx_entry;
    int idx_in_worldcoord = 0;
    int idx_out_feature = 0;
    for (int i=p.in_worldcoord_ndim-2; i>=0; i--) {
        int idx_t = t % p.in_worldcoord_dims[i];
        t = t / p.in_worldcoord_dims[i];
        idx_in_worldcoord += p.in_worldcoord_strides[i] * idx_t;
        idx_out_feature += p.out_feature_strides[i] * idx_t;
    }
    if (t > 0) {
        return;
    }
    int stride_in_worldcoord = p.in_worldcoord_strides[p.in_worldcoord_ndim-1];
    int stride_out_feature = p.out_feature_strides[p.in_worldcoord_ndim-1];


    float world_coords[3];
    world_coords[0] = in_worldcoord[idx_in_worldcoord];
    world_coords[1] = in_worldcoord[idx_in_worldcoord+stride_in_worldcoord];
    world_coords[2] = in_worldcoord[idx_in_worldcoord+stride_in_worldcoord*2];

    float local_coords[3];
    int vox_coords[3];
    local_coords[0] = world_coords[0] - floorf(world_coords[0]);
    vox_coords[0] = (int)floorf(world_coords[0]);
    local_coords[1] = world_coords[1] - floorf(world_coords[1]);
    vox_coords[1] = (int)floorf(world_coords[1]);
    local_coords[2] = world_coords[2] - floorf(world_coords[2]);
    vox_coords[2] = (int)floorf(world_coords[2]);

    float interp_weight[8];
    // 0,0,0
    interp_weight[0] = (1.0f-local_coords[0])*(1.0f-local_coords[1])*(1.0f-local_coords[2]);
    // 0,0,1
    interp_weight[1] = (1.0f-local_coords[0])*(1.0f-local_coords[1])*(local_coords[2]);
    // 0,1,0
    interp_weight[2] = (1.0f-local_coords[0])*(local_coords[1])*(1.0f-local_coords[2]);
    // 0,1,1
    interp_weight[3] = (1.0f-local_coords[0])*(local_coords[1])*(local_coords[2]);
    // 1,0,0
    interp_weight[4] = (local_coords[0])*(1.0f-local_coords[1])*(1.0f-local_coords[2]);
    // 1,0,1
    interp_weight[5] = (local_coords[0])*(1.0f-local_coords[1])*(local_coords[2]);
    // 1,1,0
    interp_weight[6] = (local_coords[0])*(local_coords[1])*(1.0f-local_coords[2]);
    // 1,1,1
    interp_weight[7] = (local_coords[0])*(local_coords[1])*(local_coords[2]);

    int indices[8];
    // Hard boundary check (zero padding)
    if (isnan(world_coords[0]) || isnan(world_coords[1]) || isnan(world_coords[2])) {
        indices[0] = -1;indices[1] = -1;indices[2] = -1;indices[3] = -1;
        indices[4] = -1;indices[5] = -1;indices[6] = -1;indices[7] = -1;
    } else {
        // Clamp to boundaries
        int vox_coords_1[3];
        vox_coords_1[0] = min(max(vox_coords[0]+1, 0), p.corner_lut_dims[0]-1);
        vox_coords_1[1] = min(max(vox_coords[1]+1, 0), p.corner_lut_dims[1]-1);
        vox_coords_1[2] = min(max(vox_coords[2]+1, 0), p.corner_lut_dims[2]-1);
        vox_coords[0] = min(max(vox_coords[0], 0), p.corner_lut_dims[0]-1);
        vox_coords[1] = min(max(vox_coords[1], 0), p.corner_lut_dims[1]-1);
        vox_coords[2] = min(max(vox_coords[2], 0), p.corner_lut_dims[2]-1);
        int idx_corner_lut;
        // 000
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords[0] +
                         p.corner_lut_strides[1] * vox_coords[1] +
                         p.corner_lut_strides[2] * vox_coords[2];
        indices[0] = corner_lut_t[idx_corner_lut];
        // 001
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords[0] +
                         p.corner_lut_strides[1] * vox_coords[1] +
                         p.corner_lut_strides[2] * vox_coords_1[2];
        indices[1] = corner_lut_t[idx_corner_lut];
        // 010
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords[0] +
                         p.corner_lut_strides[1] * vox_coords_1[1] +
                         p.corner_lut_strides[2] * vox_coords[2];
        indices[2] = corner_lut_t[idx_corner_lut];
        // 011
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords[0] +
                         p.corner_lut_strides[1] * vox_coords_1[1] +
                         p.corner_lut_strides[2] * vox_coords_1[2];
        indices[3] = corner_lut_t[idx_corner_lut];
        // 100
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords_1[0] +
                         p.corner_lut_strides[1] * vox_coords[1] +
                         p.corner_lut_strides[2] * vox_coords[2];
        indices[4] = corner_lut_t[idx_corner_lut];
        // 101
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords_1[0] +
                         p.corner_lut_strides[1] * vox_coords[1] +
                         p.corner_lut_strides[2] * vox_coords_1[2];
        indices[5] = corner_lut_t[idx_corner_lut];
        // 110
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords_1[0] +
                         p.corner_lut_strides[1] * vox_coords_1[1] +
                         p.corner_lut_strides[2] * vox_coords[2];
        indices[6] = corner_lut_t[idx_corner_lut];
        // 111
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords_1[0] +
                         p.corner_lut_strides[1] * vox_coords_1[1] +
                         p.corner_lut_strides[2] * vox_coords_1[2];
        indices[7] = corner_lut_t[idx_corner_lut];
    }

    if (p.ign_zero) {
        // Zero indices are to be ignored
#pragma unroll
        for (int i=0; i<8; i++) {
            indices[i] -= 1;
        }
    }

    //int idx_feat = blockIdx.x * TILE_DIM_X * DUP_X + threadIdx.x;
    int idx_feat = blockIdx.y * TILE_DIM_X + threadIdx.x;
    for (int i=0; i<DUP_X; i++) {
        if (idx_feat >= p.in_feature_dim) {
            return;
        }
        float interp_feat = 0.0f;
#pragma unroll
        for (int j=0; j<8; j++) {
            if (indices[j] >= 0) {
                interp_feat = fmaf(in_feature[indices[j]*p.in_feature_dim+idx_feat], interp_weight[j], interp_feat);
            }
        }
        //out_feature[idx_entry*p.in_feature_dim+idx_feat] = interp_feat;
        out_feature[idx_out_feature+stride_out_feature*idx_feat] = interp_feat;
        //idx_feat += TILE_DIM_X;
        idx_feat += TILE_DIM_X * GRID_X;
    }
}


//sp_trilinear_worldcoord_backward2feature_kernel<TILE_DIM_X, TILE_DIM_Y, DUP_X><<<dimGrid, dimBlock, 0, stream>>>(
//        in_feature_grad.data_ptr<float>(),
//        out_feature_grad.data_ptr<float>(), in_feature.data_ptr<float>(), in_corner_lut.data_ptr<int32_t>(), in_worldcoord.data_ptr<float>(), p
// Backward to feature
template <int TILE_DIM_X, int TILE_DIM_Y, int DUP_X>
__global__ void sp_trilinear_worldcoord_backward2feature_kernel(
    float* __restrict__ in_feature_grad,
    const float* __restrict__ out_feature_grad, const int32_t* __restrict__ corner_lut_t, const float* __restrict__ in_worldcoord, SpTrilinear_wc_Params p) {

    const int GRID_X = gridDim.x;
    int idx_entry = blockIdx.y * TILE_DIM_Y + threadIdx.y;

    // Index processing
    //int index[7];
    int t = idx_entry;
    int idx_in_worldcoord = 0;
    int idx_out_feature = 0;
    for (int i=p.in_worldcoord_ndim-2; i>=0; i--) {
        int idx_t = t % p.in_worldcoord_dims[i];
        t = t / p.in_worldcoord_dims[i];
        //index[i] = idx_t;
        idx_in_worldcoord += p.in_worldcoord_strides[i] * idx_t;
        idx_out_feature += p.out_feature_strides[i] * idx_t;
    }
    if (t > 0) {
        return;
    }
    int stride_in_worldcoord = p.in_worldcoord_strides[p.in_worldcoord_ndim-1];
    int stride_out_feature = p.out_feature_strides[p.in_worldcoord_ndim-1];

    float world_coords[3];
    world_coords[0] = in_worldcoord[idx_in_worldcoord];
    world_coords[1] = in_worldcoord[idx_in_worldcoord+stride_in_worldcoord];
    world_coords[2] = in_worldcoord[idx_in_worldcoord+stride_in_worldcoord*2];

    float local_coords[3];
    int vox_coords[3];
    local_coords[0] = world_coords[0] - floorf(world_coords[0]);
    vox_coords[0] = (int)floorf(world_coords[0]);
    local_coords[1] = world_coords[1] - floorf(world_coords[1]);
    vox_coords[1] = (int)floorf(world_coords[1]);
    local_coords[2] = world_coords[2] - floorf(world_coords[2]);
    vox_coords[2] = (int)floorf(world_coords[2]);

    float interp_weight[8];
    // 0,0,0
    interp_weight[0] = (1.0f-local_coords[0])*(1.0f-local_coords[1])*(1.0f-local_coords[2]);
    // 0,0,1
    interp_weight[1] = (1.0f-local_coords[0])*(1.0f-local_coords[1])*(local_coords[2]);
    // 0,1,0
    interp_weight[2] = (1.0f-local_coords[0])*(local_coords[1])*(1.0f-local_coords[2]);
    // 0,1,1
    interp_weight[3] = (1.0f-local_coords[0])*(local_coords[1])*(local_coords[2]);
    // 1,0,0
    interp_weight[4] = (local_coords[0])*(1.0f-local_coords[1])*(1.0f-local_coords[2]);
    // 1,0,1
    interp_weight[5] = (local_coords[0])*(1.0f-local_coords[1])*(local_coords[2]);
    // 1,1,0
    interp_weight[6] = (local_coords[0])*(local_coords[1])*(1.0f-local_coords[2]);
    // 1,1,1
    interp_weight[7] = (local_coords[0])*(local_coords[1])*(local_coords[2]);

    int indices[8];
    // Hard boundary check (zero padding)
    if (isnan(world_coords[0]) || isnan(world_coords[1]) || isnan(world_coords[2])) {// ||
        //vox_coords[0] < 0 || vox_coords[0] >= (p.corner_lut_dims[0]-1) ||
        //vox_coords[1] < 0 || vox_coords[1] >= (p.corner_lut_dims[1]-1) ||
        //vox_coords[2] < 0 || vox_coords[2] >= (p.corner_lut_dims[2]-1)) {
        indices[0] = -1;indices[1] = -1;indices[2] = -1;indices[3] = -1;
        indices[4] = -1;indices[5] = -1;indices[6] = -1;indices[7] = -1;
    } else {
        // Clamp to boundaries
        int vox_coords_1[3];
        vox_coords_1[0] = min(max(vox_coords[0]+1, 0), p.corner_lut_dims[0]-1);
        vox_coords_1[1] = min(max(vox_coords[1]+1, 0), p.corner_lut_dims[1]-1);
        vox_coords_1[2] = min(max(vox_coords[2]+1, 0), p.corner_lut_dims[2]-1);
        vox_coords[0] = min(max(vox_coords[0], 0), p.corner_lut_dims[0]-1);
        vox_coords[1] = min(max(vox_coords[1], 0), p.corner_lut_dims[1]-1);
        vox_coords[2] = min(max(vox_coords[2], 0), p.corner_lut_dims[2]-1);
        int idx_corner_lut;
        // 000
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords[0] +
                         p.corner_lut_strides[1] * vox_coords[1] +
                         p.corner_lut_strides[2] * vox_coords[2];
        indices[0] = corner_lut_t[idx_corner_lut];
        // 001
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords[0] +
                         p.corner_lut_strides[1] * vox_coords[1] +
                         p.corner_lut_strides[2] * vox_coords_1[2];
        indices[1] = corner_lut_t[idx_corner_lut];
        // 010
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords[0] +
                         p.corner_lut_strides[1] * vox_coords_1[1] +
                         p.corner_lut_strides[2] * vox_coords[2];
        indices[2] = corner_lut_t[idx_corner_lut];
        // 011
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords[0] +
                         p.corner_lut_strides[1] * vox_coords_1[1] +
                         p.corner_lut_strides[2] * vox_coords_1[2];
        indices[3] = corner_lut_t[idx_corner_lut];
        // 100
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords_1[0] +
                         p.corner_lut_strides[1] * vox_coords[1] +
                         p.corner_lut_strides[2] * vox_coords[2];
        indices[4] = corner_lut_t[idx_corner_lut];
        // 101
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords_1[0] +
                         p.corner_lut_strides[1] * vox_coords[1] +
                         p.corner_lut_strides[2] * vox_coords_1[2];
        indices[5] = corner_lut_t[idx_corner_lut];
        // 110
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords_1[0] +
                         p.corner_lut_strides[1] * vox_coords_1[1] +
                         p.corner_lut_strides[2] * vox_coords[2];
        indices[6] = corner_lut_t[idx_corner_lut];
        // 111
        idx_corner_lut = p.corner_lut_strides[0] * vox_coords_1[0] +
                         p.corner_lut_strides[1] * vox_coords_1[1] +
                         p.corner_lut_strides[2] * vox_coords_1[2];
        indices[7] = corner_lut_t[idx_corner_lut];
    }

    if (p.ign_zero) {
#pragma unroll
        for (int i=0; i<8; i++) {
            indices[i] -= 1;
        }
    }

    //int idx_feat = blockIdx.x * TILE_DIM_X * DUP_X + threadIdx.x;
    int idx_feat = blockIdx.x * TILE_DIM_X + threadIdx.x;
    for (int i=0; i<DUP_X; i++) {
        if (idx_feat >= p.in_feature_dim) {
            return;
        }
        float grad = out_feature_grad[idx_out_feature+stride_out_feature*idx_feat];
#pragma unroll
        for (int j=0; j<8; j++) {
            if (indices[j] >= 0) {
                //indices[j]*p.in_feature_dim+idx_feat
                atomicAdd(&in_feature_grad[indices[j]*p.in_feature_dim+idx_feat], grad * interp_weight[j]);
            }
        }
        //idx_feat += TILE_DIM_X;
        idx_feat += TILE_DIM_X * GRID_X;
    }
}

// in_feature, corner_lut_t, in_world_coord, ign_zero=False
// Input:
//      in_feature:     float32 [M C]
//      in_corner_lut:  int32   [X Y Z]
//      in_worldcoord:  float32 [..., 3]
//      ---Index:          int32   [..., 8], containing [0, M]. 0 is ignore label.
//      ---Coord:          float32 [..., 3]
// Output:
//      Interp. Feat:   float32 [..., C]
// std::vector<torch::Tensor>
torch::Tensor sp_trilinear_worldcoord_cuda(const torch::Tensor& in_feature, const torch::Tensor& in_corner_lut, const torch::Tensor& in_worldcoord, bool ign_zero, int channel_pos) {
    CHECK_CUDA(in_feature);
    CHECK_CUDA(in_corner_lut);
    CHECK_CUDA(in_worldcoord);

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    torch::Device device = in_feature.device();

    // assert(tensor.sizes() == std::vector<int64_t>{3, 4, 5});
    assert(in_feature.dtype() == torch::kFloat32);
    assert(in_feature.dim() == 2);
    assert(in_corner_lut.dtype() == torch::kInt32);
    assert(in_corner_lut.dim() == 3);
    assert(in_worldcoord.dtype() == torch::kFloat32);
    assert(in_worldcoord.size(-1) == 3);
    assert(in_worldcoord.dim() <= 8);

    CHECK_CONTIGUOUS(in_feature);
    //CHECK_CONTIGUOUS(in_corner_lut); // Will still run correctly, but performance will suffer.
    //CHECK_CONTIGUOUS(in_worldcoord);

    //int channel_pos = -1; // -1 for HWC, -3 for CHW
    if (channel_pos < 0) {
        channel_pos += in_worldcoord.dim();
    }
    assert(channel_pos >= 0 && channel_pos < in_worldcoord.dim());

    SpTrilinear_wc_Params p;
    p.in_feature_dim = in_feature.size(1);
    p.in_feature_numentries = in_feature.size(0);
    p.in_worldcoord_ndim = in_worldcoord.dim();
    for (int i=0; i<in_worldcoord.dim(); i++) {
        p.in_worldcoord_dims[i] = in_worldcoord.size(i);
        p.in_worldcoord_strides[i] = in_worldcoord.stride(i);
    }
    p.ign_zero = ign_zero;

    p.corner_lut_dims[0] = in_corner_lut.size(0);
    p.corner_lut_dims[1] = in_corner_lut.size(1);
    p.corner_lut_dims[2] = in_corner_lut.size(2);
    p.corner_lut_strides[0] = in_corner_lut.stride(0);
    p.corner_lut_strides[1] = in_corner_lut.stride(1);
    p.corner_lut_strides[2] = in_corner_lut.stride(2);

    int numentries = in_worldcoord.numel() / 3;
    //printf("FWD numentries: %d\n", numentries);

    std::vector<int64_t> out_feature_shape;
    //if (channel_first) { // Channel First format, suitable for 2D convolution
    //    //assert(false);
    for (int i=0; i<channel_pos; i++) {
        out_feature_shape.push_back(in_worldcoord.size(i));
    }
    out_feature_shape.push_back(p.in_feature_dim);
    for (int i=channel_pos; i<in_worldcoord.dim()-1; i++) {
        out_feature_shape.push_back(in_worldcoord.size(i));
    }
    torch::Tensor out_feature = torch::empty(out_feature_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    // The feature is always at the last dimension. Swap it to the last dim.
    for (int i=channel_pos+1; i<out_feature.dim(); i++) {
        out_feature.transpose_(i-1, i);
    }
    //} else { // Channel Last
    //    for (int i=0; i<in_worldcoord.dim()-1; i++) {
    //        out_feature_shape.push_back(in_worldcoord.size(i));
    //    }
    //    out_feature_shape.push_back(p.in_feature_dim);
    //    out_feature = torch::empty(out_feature_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    //}
    for (int i=0; i<out_feature.dim(); i++) {
        p.out_feature_dims[i] = out_feature.size(i);
        p.out_feature_strides[i] = out_feature.stride(i);
    }

    const int TILE_DIM_X = 16;  // feature dim
    const int TILE_DIM_Y = 64;  // entry dim
    const int DUP_X = 4;   // To amortize the cost of weight computation
    //dim3 dimGrid((p.in_feature_dim+(TILE_DIM_X*DUP_X)-1)/(TILE_DIM_X*DUP_X), (numentries+TILE_DIM_Y-1)/TILE_DIM_Y, 1);
    dim3 dimGrid((numentries+TILE_DIM_Y-1)/TILE_DIM_Y, (p.in_feature_dim+(TILE_DIM_X*DUP_X)-1)/(TILE_DIM_X*DUP_X), 1);
    dim3 dimBlock(TILE_DIM_X, TILE_DIM_Y, 1);

    sp_trilinear_worldcoord_kernel<TILE_DIM_X, TILE_DIM_Y, DUP_X><<<dimGrid, dimBlock, 0, stream>>>(
        out_feature.data_ptr<float>(),
        in_feature.data_ptr<float>(), in_corner_lut.data_ptr<int32_t>(), in_worldcoord.data_ptr<float>(), p
    );
    THCudaCheck(cudaGetLastError());
    return out_feature;
}


// Backward function for sparse trilinear interpolation
// Input:
//      out_feature_grad:   float32 [..., C]
//      in_feature:         float32 [M, C]
//      in_corner_lut:      int32   [X Y Z]
//      ---in_index:        int32   [..., 8], containing [0, M]. 0 is ignore label.
//      in_worldcoord:      float32 [..., 3]
//      ign_zero:           bool
//      need_coord_grad:    bool
// Output:
//      in_feature_grad:    float32 [M, C]
//      in_coord_grad:      float32 [..., 3]
std::vector<torch::Tensor> sp_trilinear_worldcoord_backward_cuda(const torch::Tensor& out_feature_grad , const torch::Tensor& in_feature, const torch::Tensor& in_corner_lut, const torch::Tensor& in_worldcoord, bool ign_zero, bool need_coord_grad) {
    assert(need_coord_grad == false);
    CHECK_CUDA(out_feature_grad);
    CHECK_CUDA(in_feature);
    CHECK_CUDA(in_corner_lut);
    CHECK_CUDA(in_worldcoord);

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    torch::Device device = out_feature_grad.device();

    //for (int i=0; i<out_feature_grad.dim(); i++) {
    //    printf("[sp_trilinear_backward_cuda] dim, size, stride: %d, %d, %d\n", i, out_feature_grad.size(i), out_feature_grad.stride(i));
    //}
    //CHECK_CONTIGUOUS(out_feature_grad);
    CHECK_CONTIGUOUS(in_feature);
    //CHECK_CONTIGUOUS(in_worldcoord);

    // assert(tensor.sizes() == std::vector<int64_t>{3, 4, 5});
    assert(out_feature_grad.dtype() == torch::kFloat32);
    for (int i=0; i<out_feature_grad.dim()-1; i++) {
        assert(out_feature_grad.size(i) == in_worldcoord.size(i));
    }
    assert(out_feature_grad.size(-1) == in_feature.size(1));
    assert(in_feature.dtype() == torch::kFloat32);
    assert(in_feature.dim() == 2);
    assert(in_worldcoord.dtype() == torch::kFloat32);
    assert(in_worldcoord.size(-1) == 3);

    SpTrilinear_wc_Params p;
    p.in_feature_dim = in_feature.size(1);
    p.in_feature_numentries = in_feature.size(0);
    p.in_worldcoord_ndim = in_worldcoord.dim();
    for (int i=0; i<in_worldcoord.dim(); i++) {
        p.in_worldcoord_dims[i] = in_worldcoord.size(i);
        p.in_worldcoord_strides[i] = in_worldcoord.stride(i);
    }
    p.ign_zero = ign_zero;

    p.corner_lut_dims[0] = in_corner_lut.size(0);
    p.corner_lut_dims[1] = in_corner_lut.size(1);
    p.corner_lut_dims[2] = in_corner_lut.size(2);
    p.corner_lut_strides[0] = in_corner_lut.stride(0);
    p.corner_lut_strides[1] = in_corner_lut.stride(1);
    p.corner_lut_strides[2] = in_corner_lut.stride(2);

    for (int i=0; i<out_feature_grad.dim(); i++) {
        p.out_feature_dims[i] = out_feature_grad.size(i);
        p.out_feature_strides[i] = out_feature_grad.stride(i);
    }
    int numentries = in_worldcoord.numel() / 3;

    // Create output tensors
    torch::Tensor in_feature_grad = torch::zeros({p.in_feature_numentries, p.in_feature_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor in_coord_grad;

    {
        const int TILE_DIM_X = 16;  // feature dim
        const int TILE_DIM_Y = 64;  // entry dim
        const int DUP_X = 4;   // To amortize the cost of weight computation
        dim3 dimGrid((p.in_feature_dim+(TILE_DIM_X*DUP_X)-1)/(TILE_DIM_X*DUP_X), (numentries+TILE_DIM_Y-1)/TILE_DIM_Y, 1);
        dim3 dimBlock(TILE_DIM_X, TILE_DIM_Y, 1);
        //printf("BW dimGrid: %d, %d, %d \n", dimGrid.x, dimGrid.y, dimGrid.z);
        sp_trilinear_worldcoord_backward2feature_kernel<TILE_DIM_X, TILE_DIM_Y, DUP_X><<<dimGrid, dimBlock, 0, stream>>>(
            in_feature_grad.data_ptr<float>(),
            out_feature_grad.data_ptr<float>(), in_corner_lut.data_ptr<int32_t>(), in_worldcoord.data_ptr<float>(), p
        );
    }

    THCudaCheck(cudaGetLastError());
    return {in_feature_grad};
}
