// Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, check out LICENSE.md
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

// Fast voxel traversal along rays
std::vector<torch::Tensor> ray_voxel_intersection_perspective_cuda(const torch::Tensor& in_voxel, const torch::Tensor& cam_ori, const torch::Tensor& cam_dir, const torch::Tensor& cam_up, float cam_f, const std::vector<float>& cam_c, const std::vector<int>& img_dims, int max_samples);


// World Coordinate Sparse Trilinear Interpolation
torch::Tensor sp_trilinear_worldcoord_cuda(const torch::Tensor& in_feature, const torch::Tensor& in_corner_lut, const torch::Tensor& in_worldcoord, bool ign_zero, int channel_pos);

std::vector<torch::Tensor> sp_trilinear_worldcoord_backward_cuda(const torch::Tensor& out_feature_grad , const torch::Tensor& in_feature, const torch::Tensor& in_corner_lut, const torch::Tensor& in_worldcoord, bool ign_zero, bool need_coord_grad);

// Fast & Memory Efficient Positional Encoding
torch::Tensor positional_encoding_cuda(const torch::Tensor& in_feature, int ndegrees, int dim, bool incl_orig);

torch::Tensor positional_encoding_backward_cuda(const torch::Tensor& out_feature_grad, const torch::Tensor& out_feature, int ndegrees, int dim, bool incl_orig);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ray_voxel_intersection_perspective", &ray_voxel_intersection_perspective_cuda, "Ray-voxel intersections given perspective camera parameters (CUDA)");
    m.def("sp_trilinear_worldcoord", &sp_trilinear_worldcoord_cuda, "Sparse Trilinear interpolation, world coordinate [forward] (CUDA)");
    m.def("sp_trilinear_worldcoord_backward", &sp_trilinear_worldcoord_backward_cuda, "Sparse Trilinear interpolation, world coordinate [backward] (CUDA)");
    m.def("positional_encoding", &positional_encoding_cuda, "Fused Positional Encoding [forward] (CUDA)");
    m.def("positional_encoding_backward", &positional_encoding_backward_cuda, "Fused Positional Encoding [backward] (CUDA)");
}