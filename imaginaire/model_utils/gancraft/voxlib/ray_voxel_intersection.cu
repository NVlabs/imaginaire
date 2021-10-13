// Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, check out LICENSE.md
//
// The ray marching algorithm used in this file is a variety of modified Bresenham method:
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
// Search for "voxel traversal algorithm" for related information

#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

//#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "voxlib_common.h"

struct RVIP_Params {
    int voxel_dims[3];
    int voxel_strides[3];
    int max_samples;
    int img_dims[2];
    // Camera parameters
    float cam_ori[3];
    float cam_fwd[3];
    float cam_side[3];
    float cam_up[3];
    float cam_c[2];
    float cam_f;
    //unsigned long seed;
};

/*
    out_voxel_id: torch CUDA int32  [   img_dims[0], img_dims[1], max_samples, 1]
    out_depth:    torch CUDA float  [2, img_dims[0], img_dims[1], max_samples, 1]
    out_raydirs:  torch CUDA float  [   img_dims[0], img_dims[1],           1, 3]
    Image coordinates refer to the center of the pixel
    [0, 0, 0] at voxel coordinate is at the corner of the corner block (instead of at the center)
*/
template <int TILE_DIM>
static __global__ void ray_voxel_intersection_perspective_kernel(int32_t* __restrict__ out_voxel_id, float* __restrict__ out_depth, float* __restrict__ out_raydirs,
const int32_t* __restrict__ in_voxel, const RVIP_Params p) {

    int img_coords[2];
    img_coords[1] = blockIdx.x*TILE_DIM+threadIdx.x;
    img_coords[0] = blockIdx.y*TILE_DIM+threadIdx.y;
    if (img_coords[0] >= p.img_dims[0] || img_coords[1] >= p.img_dims[1]) {
        return;
    }
    int pix_index = img_coords[0] * p.img_dims[1] + img_coords[1];

    // Calculate ray origin and direction
    float rayori[3], raydir[3];
    rayori[0] = p.cam_ori[0];
    rayori[1] = p.cam_ori[1];
    rayori[2] = p.cam_ori[2];

    // Camera intrinsics
    float ndc_imcoords[2];
    ndc_imcoords[0] = p.cam_c[0] - (float)img_coords[0]; // Flip height
    ndc_imcoords[1] = (float)img_coords[1] - p.cam_c[1];

    raydir[0] = p.cam_up[0] * ndc_imcoords[0] + p.cam_side[0] * ndc_imcoords[1] + p.cam_fwd[0] * p.cam_f;
    raydir[1] = p.cam_up[1] * ndc_imcoords[0] + p.cam_side[1] * ndc_imcoords[1] + p.cam_fwd[1] * p.cam_f;
    raydir[2] = p.cam_up[2] * ndc_imcoords[0] + p.cam_side[2] * ndc_imcoords[1] + p.cam_fwd[2] * p.cam_f;
    normalize<float, 3>(raydir);

    // Save out_raydirs
    out_raydirs[pix_index*3] = raydir[0];
    out_raydirs[pix_index*3+1] = raydir[1];
    out_raydirs[pix_index*3+2] = raydir[2];

    float axis_t[3];
    int axis_int[3];
    //int axis_intbound[3];

    // Current voxel
    axis_int[0] = floorf(rayori[0]);
    axis_int[1] = floorf(rayori[1]);
    axis_int[2] = floorf(rayori[2]);

    #pragma unroll
    for (int i=0; i<3; i++) {
        if (raydir[i] > 0) {
            // Initial t value
            // Handle boundary case where rayori[i] is a whole number. Always round Up for the next block
            //axis_t[i] = (ceilf(nextafterf(rayori[i], HUGE_VALF)) - rayori[i]) / raydir[i];
            axis_t[i] = ((float)(axis_int[i]+1) - rayori[i]) / raydir[i];
        } else if (raydir[i] < 0) {
            axis_t[i] = ((float)axis_int[i] - rayori[i]) / raydir[i];
        } else {
            axis_t[i] = HUGE_VALF;
        }
    }

    // Fused raymarching and sampling
    bool quit = false;
    for (int cur_plane=0; cur_plane < p.max_samples; cur_plane++) { // Last cycle is for calculating p2
        float t = nanf("0");
        float t2 = nanf("0");
        int32_t blk_id = 0;
        // Find the next intersection
        while (!quit) {
            // Find the next smallest t
            float tnow;
            /*
            #pragma unroll
            for (int i=0; i<3; i++) {
                if (axis_t[i] <= axis_t[(i+1)%3] && axis_t[i] <= axis_t[(i+2)%3]) {
                    // Update current t
                    tnow = axis_t[i];
                    // Update t candidates
                    if (raydir[i] > 0) {
                        axis_int[i] += 1;
                        if (axis_int[i] >= p.voxel_dims[i]) {
                            quit = true;
                        }
                        axis_t[i] = ((float)(axis_int[i]+1) - rayori[i]) / raydir[i];
                    } else {
                        axis_int[i] -= 1;
                        if (axis_int[i] < 0) {
                            quit = true;
                        }
                        axis_t[i] = ((float)axis_int[i] - rayori[i]) / raydir[i];
                    }
                    break; // Avoid advancing multiple steps as axis_t is updated
                }
            }
            */
            // Hand unroll
            if (axis_t[0] <= axis_t[1] && axis_t[0] <= axis_t[2]) {
                // Update current t
                tnow = axis_t[0];
                // Update t candidates
                if (raydir[0] > 0) {
                    axis_int[0] += 1;
                    if (axis_int[0] >= p.voxel_dims[0]) {
                        quit = true;
                    }
                    axis_t[0] = ((float)(axis_int[0]+1) - rayori[0]) / raydir[0];
                } else {
                    axis_int[0] -= 1;
                    if (axis_int[0] < 0) {
                        quit = true;
                    }
                    axis_t[0] = ((float)axis_int[0] - rayori[0]) / raydir[0];
                }
            } else if (axis_t[1] <= axis_t[2]) {
                tnow = axis_t[1];
                if (raydir[1] > 0) {
                    axis_int[1] += 1;
                    if (axis_int[1] >= p.voxel_dims[1]) {
                        quit = true;
                    }
                    axis_t[1] = ((float)(axis_int[1]+1) - rayori[1]) / raydir[1];
                } else {
                    axis_int[1] -= 1;
                    if (axis_int[1] < 0) {
                        quit = true;
                    }
                    axis_t[1] = ((float)axis_int[1] - rayori[1]) / raydir[1];
                }
            } else {
                tnow = axis_t[2];
                if (raydir[2] > 0) {
                    axis_int[2] += 1;
                    if (axis_int[2] >= p.voxel_dims[2]) {
                        quit = true;
                    }
                    axis_t[2] = ((float)(axis_int[2]+1) - rayori[2]) / raydir[2];
                } else {
                    axis_int[2] -= 1;
                    if (axis_int[2] < 0) {
                        quit = true;
                    }
                    axis_t[2] = ((float)axis_int[2] - rayori[2]) / raydir[2];
                }
            }

            if (quit) {
                break;
            }

            // Skip empty space
            // Could there be deadlock if the ray direction is away from the world?
            if (axis_int[0] < 0 || axis_int[0] >= p.voxel_dims[0] || axis_int[1] < 0 || axis_int[1] >= p.voxel_dims[1] || axis_int[2] < 0 || axis_int[2] >= p.voxel_dims[2]) {
                continue;
            }

            // Test intersection using voxel grid
            blk_id = in_voxel[axis_int[0]*p.voxel_strides[0] + axis_int[1]*p.voxel_strides[1] + axis_int[2]*p.voxel_strides[2]];
            if (blk_id == 0) {
                continue;
            }

            // Now that there is an intersection
            t = tnow;
            // Calculate t2
            /*
            #pragma unroll
            for (int i=0; i<3; i++) {
                if (axis_t[i] <= axis_t[(i+1)%3] && axis_t[i] <= axis_t[(i+2)%3]) {
                    t2 = axis_t[i];
                    break;
                }
            }
            */
            // Hand unroll
            if (axis_t[0] <= axis_t[1] && axis_t[0] <= axis_t[2]) {
                t2 = axis_t[0];
            } else if (axis_t[1] <= axis_t[2]) {
                t2 = axis_t[1];
            } else {
                t2 = axis_t[2];
            }
            break;
        } // while !quit (ray marching loop)

        out_depth[pix_index*p.max_samples+cur_plane] = t;
        out_depth[p.img_dims[0]*p.img_dims[1]*p.max_samples + pix_index*p.max_samples+cur_plane] = t2;
        out_voxel_id[pix_index*p.max_samples+cur_plane] = blk_id;
    } // cur_plane
}


/*
    out:
        out_voxel_id: torch CUDA int32  [   img_dims[0], img_dims[1], max_samples, 1]
        out_depth:    torch CUDA float  [2, img_dims[0], img_dims[1], max_samples, 1]
        out_raydirs:  torch CUDA float  [   img_dims[0], img_dims[1],           1, 3]
    in:
        in_voxel:     torch CUDA int32  [X, Y, Z] [40, 512, 512]
        cam_ori:      torch      float  [3]
        cam_dir:      torch      float  [3]
        cam_up:       torch      float  [3]
        cam_f:                   float
        cam_c:                   int    [2]
        img_dims:                int    [2]
        max_samples:             int
*/
std::vector<torch::Tensor> ray_voxel_intersection_perspective_cuda(const torch::Tensor& in_voxel, const torch::Tensor& cam_ori, const torch::Tensor& cam_dir, const torch::Tensor& cam_up, float cam_f, const std::vector<float>& cam_c, const std::vector<int>& img_dims, int max_samples) {
    CHECK_CUDA(in_voxel);

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    torch::Device device = in_voxel.device();

    //assert(in_voxel.dtype() == torch::kU8);
    assert(in_voxel.dtype() == torch::kInt32); // Minecraft compatibility
    assert(in_voxel.dim() == 3);
    assert(cam_ori.dtype() == torch::kFloat32);
    assert(cam_ori.numel() == 3);
    assert(cam_dir.dtype() == torch::kFloat32);
    assert(cam_dir.numel() == 3);
    assert(cam_up.dtype() == torch::kFloat32);
    assert(cam_up.numel() == 3);
    assert(img_dims.size() == 2);

    RVIP_Params p;

    // Calculate camera rays
    const torch::Tensor cam_ori_c = cam_ori.cpu();
    const torch::Tensor cam_dir_c = cam_dir.cpu();
    const torch::Tensor cam_up_c = cam_up.cpu();

    // Get the coordinate frame of camera space in world space
    normalize<float, 3>(p.cam_fwd, cam_dir_c.data_ptr<float>());
    cross<float>(p.cam_side, p.cam_fwd, cam_up_c.data_ptr<float>());
    normalize<float, 3>(p.cam_side);
    cross<float>(p.cam_up, p.cam_side, p.cam_fwd);
    normalize<float, 3>(p.cam_up); // Not absolutely necessary as both vectors are normalized. But just in case...

    copyarr<float, 3>(p.cam_ori, cam_ori_c.data_ptr<float>());

    p.cam_f = cam_f;
    p.cam_c[0] = cam_c[0];
    p.cam_c[1] = cam_c[1];
    p.max_samples = max_samples;
    //printf("[Renderer] max_dist: %ld\n", max_dist);

    p.voxel_dims[0] = in_voxel.size(0);
    p.voxel_dims[1] = in_voxel.size(1);
    p.voxel_dims[2] = in_voxel.size(2);
    p.voxel_strides[0] = in_voxel.stride(0);
    p.voxel_strides[1] = in_voxel.stride(1);
    p.voxel_strides[2] = in_voxel.stride(2);

    //printf("[Renderer] Voxel resolution: %ld, %ld, %ld\n", p.voxel_dims[0], p.voxel_dims[1], p.voxel_dims[2]);

    p.img_dims[0] = img_dims[0];
    p.img_dims[1] = img_dims[1];

    // Create output tensors
    // For Minecraft Seg Mask
    torch::Tensor out_voxel_id = torch::empty({p.img_dims[0], p.img_dims[1], p.max_samples, 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    torch::Tensor out_depth;
    // Produce two sets of localcoords, one for entry point, the other one for exit point. They share the same corner_ids.
    out_depth = torch::empty({2, p.img_dims[0], p.img_dims[1], p.max_samples, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor out_raydirs = torch::empty({p.img_dims[0], p.img_dims[1], 1, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(false));

    const int TILE_DIM = 8;
    dim3 dimGrid((p.img_dims[1]+TILE_DIM-1)/TILE_DIM, (p.img_dims[0]+TILE_DIM-1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);

    ray_voxel_intersection_perspective_kernel<TILE_DIM><<<dimGrid, dimBlock, 0, stream>>>(
        out_voxel_id.data_ptr<int32_t>(), out_depth.data_ptr<float>(), out_raydirs.data_ptr<float>(), in_voxel.data_ptr<int32_t>(), p
    );

    return {out_voxel_id, out_depth, out_raydirs};
}
