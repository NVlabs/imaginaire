# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imaginaire.layers import Conv2dBlock, LinearBlock
from imaginaire.model_utils.gancraft.layers import AffineMod, ModLinear
import imaginaire.model_utils.gancraft.mc_utils as mc_utils
import imaginaire.model_utils.gancraft.voxlib as voxlib
from imaginaire.utils.distributed import master_only_print as print


class RenderMLP(nn.Module):
    r""" MLP with affine modulation."""

    def __init__(self, in_channels, style_dim, viewdir_dim, mask_dim=680,
                 out_channels_s=1, out_channels_c=3, hidden_channels=256,
                 use_seg=True):
        super(RenderMLP, self).__init__()

        self.use_seg = use_seg
        if self.use_seg:
            self.fc_m_a = nn.Linear(mask_dim, hidden_channels, bias=False)

        self.fc_viewdir = None
        if viewdir_dim > 0:
            self.fc_viewdir = nn.Linear(viewdir_dim, hidden_channels, bias=False)

        self.fc_1 = nn.Linear(in_channels, hidden_channels)

        self.fc_2 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_3 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_4 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)

        self.fc_sigma = nn.Linear(hidden_channels, out_channels_s)

        if viewdir_dim > 0:
            self.fc_5 = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.mod_5 = AffineMod(hidden_channels, style_dim, mod_bias=True)
        else:
            self.fc_5 = ModLinear(hidden_channels, hidden_channels, style_dim,
                                  bias=False, mod_bias=True, output_mode=True)
        self.fc_6 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_out_c = nn.Linear(hidden_channels, out_channels_c)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, raydir, z, m):
        r""" Forward network

        Args:
            x (N x H x W x M x in_channels tensor): Projected features.
            raydir (N x H x W x 1 x viewdir_dim tensor): Ray directions.
            z (N x style_dim tensor): Style codes.
            m (N x H x W x M x mask_dim tensor): One-hot segmentation maps.
        """
        b, h, w, n, _ = x.size()
        z = z[:, None, None, None, :]

        f = self.fc_1(x)
        if self.use_seg:
            f = f + self.fc_m_a(m)
        # Common MLP
        f = self.act(f)
        f = self.act(self.fc_2(f, z))
        f = self.act(self.fc_3(f, z))
        f = self.act(self.fc_4(f, z))

        # Sigma MLP
        sigma = self.fc_sigma(f)

        # Color MLP
        if self.fc_viewdir is not None:
            f = self.fc_5(f)
            f = f + self.fc_viewdir(raydir)
            f = self.act(self.mod_5(f, z))
        else:
            f = self.act(self.fc_5(f, z))
        f = self.act(self.fc_6(f, z))
        c = self.fc_out_c(f)
        return sigma, c


class StyleMLP(nn.Module):
    r"""MLP converting style code to intermediate style representation."""

    def __init__(self, style_dim, out_dim, hidden_channels=256, leaky_relu=True, num_layers=5, normalize_input=True,
                 output_act=True):
        super(StyleMLP, self).__init__()

        self.normalize_input = normalize_input
        self.output_act = output_act
        fc_layers = []
        fc_layers.append(nn.Linear(style_dim, hidden_channels, bias=True))
        for i in range(num_layers-1):
            fc_layers.append(nn.Linear(hidden_channels, hidden_channels, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)

        self.fc_out = nn.Linear(hidden_channels, out_dim, bias=True)

        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = functools.partial(F.relu, inplace=True)

    def forward(self, z):
        r""" Forward network

        Args:
            z (N x style_dim tensor): Style codes.
        """
        if self.normalize_input:
            z = F.normalize(z, p=2, dim=-1)
        for fc_layer in self.fc_layers:
            z = self.act(fc_layer(z))
        z = self.fc_out(z)
        if self.output_act:
            z = self.act(z)
        return z


class SKYMLP(nn.Module):
    r"""MLP converting ray directions to sky features."""

    def __init__(self, in_channels, style_dim, out_channels_c=3,
                 hidden_channels=256, leaky_relu=True):
        super(SKYMLP, self).__init__()
        self.fc_z_a = nn.Linear(style_dim, hidden_channels, bias=False)

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, hidden_channels)
        self.fc5 = nn.Linear(hidden_channels, hidden_channels)

        self.fc_out_c = nn.Linear(hidden_channels, out_channels_c)

        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = functools.partial(F.relu, inplace=True)

    def forward(self, x, z):
        r"""Forward network

        Args:
            x (... x in_channels tensor): Ray direction embeddings.
            z (... x style_dim tensor): Style codes.
        """

        z = self.fc_z_a(z)
        while z.dim() < x.dim():
            z = z.unsqueeze(1)

        y = self.act(self.fc1(x) + z)
        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))
        y = self.act(self.fc4(y))
        y = self.act(self.fc5(y))
        c = self.fc_out_c(y)

        return c


class RenderCNN(nn.Module):
    r"""CNN converting intermediate feature map to final image."""

    def __init__(self, in_channels, style_dim, hidden_channels=256,
                 leaky_relu=True):
        super(RenderCNN, self).__init__()
        self.fc_z_cond = nn.Linear(style_dim, 2 * 2 * hidden_channels)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv2a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)

        self.conv3a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)

        self.conv4a = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv4b = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)

        self.conv4 = nn.Conv2d(hidden_channels, 3, 1, stride=1, padding=0)

        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = functools.partial(F.relu, inplace=True)

    def modulate(self, x, w, b):
        w = w[..., None, None]
        b = b[..., None, None]
        return x * (w+1) + b

    def forward(self, x, z):
        r"""Forward network.

        Args:
            x (N x in_channels x H x W tensor): Intermediate feature map
            z (N x style_dim tensor): Style codes.
        """
        z = self.fc_z_cond(z)
        adapt = torch.chunk(z, 2 * 2, dim=-1)

        y = self.act(self.conv1(x))

        y = y + self.conv2b(self.act(self.conv2a(y)))
        y = self.act(self.modulate(y, adapt[0], adapt[1]))

        y = y + self.conv3b(self.act(self.conv3a(y)))
        y = self.act(self.modulate(y, adapt[2], adapt[3]))

        y = y + self.conv4b(self.act(self.conv4a(y)))
        y = self.act(y)

        y = self.conv4(y)

        return y


class StyleEncoder(nn.Module):
    r"""Style Encoder constructor.

    Args:
        style_enc_cfg (obj): Style encoder definition file.
    """

    def __init__(self, style_enc_cfg):
        super(StyleEncoder, self).__init__()
        input_image_channels = style_enc_cfg.input_image_channels
        num_filters = style_enc_cfg.num_filters
        kernel_size = style_enc_cfg.kernel_size
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        style_dims = style_enc_cfg.style_dims
        weight_norm_type = style_enc_cfg.weight_norm_type
        self.no_vae = getattr(style_enc_cfg, 'no_vae', False)
        activation_norm_type = 'none'
        nonlinearity = 'leakyrelu'
        base_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              # inplace_nonlinearity=True,
                              nonlinearity=nonlinearity)
        self.layer1 = base_conv2d_block(input_image_channels, num_filters)
        self.layer2 = base_conv2d_block(num_filters * 1, num_filters * 2)
        self.layer3 = base_conv2d_block(num_filters * 2, num_filters * 4)
        self.layer4 = base_conv2d_block(num_filters * 4, num_filters * 8)
        self.layer5 = base_conv2d_block(num_filters * 8, num_filters * 8)
        self.layer6 = base_conv2d_block(num_filters * 8, num_filters * 8)
        self.fc_mu = LinearBlock(num_filters * 8 * 4 * 4, style_dims)
        if not self.no_vae:
            self.fc_var = LinearBlock(num_filters * 8 * 4 * 4, style_dims)

    def forward(self, input_x):
        r"""SPADE Style Encoder forward.

        Args:
            input_x (N x 3 x H x W tensor): input images.
        Returns:
            mu (N x C tensor): Mean vectors.
            logvar (N x C tensor): Log-variance vectors.
            z (N x C tensor): Style code vectors.
        """
        if input_x.size(2) != 256 or input_x.size(3) != 256:
            input_x = F.interpolate(input_x, size=(256, 256), mode='bilinear')
        x = self.layer1(input_x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        if not self.no_vae:
            logvar = self.fc_var(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else:
            z = mu
            logvar = torch.zeros_like(mu)
        return mu, logvar, z


class Base3DGenerator(nn.Module):
    r"""Minecraft 3D generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super(Base3DGenerator, self).__init__()
        print('Base3DGenerator initialization.')

        # ---------------------- Main Network ------------------------
        # Exclude some of the features from positional encoding
        self.pe_no_pe_feat_dim = getattr(gen_cfg, 'pe_no_pe_feat_dim', 0)

        # blk_feat passes through PE
        input_dim = (gen_cfg.blk_feat_dim-self.pe_no_pe_feat_dim)*(gen_cfg.pe_lvl_feat*2) + self.pe_no_pe_feat_dim
        if (gen_cfg.pe_incl_orig_feat):
            input_dim += (gen_cfg.blk_feat_dim-self.pe_no_pe_feat_dim)
        print('[Base3DGenerator] Expected input dimensions: ', input_dim)
        self.input_dim = input_dim

        self.mlp_model_kwargs = gen_cfg.mlp_model_kwargs
        self.pe_lvl_localcoords = getattr(gen_cfg, 'pe_lvl_localcoords', 0)
        if self.pe_lvl_localcoords > 0:
            self.mlp_model_kwargs['poscode_dim'] = self.pe_lvl_localcoords * 2 * 3

        # Set pe_lvl_raydir=0 and pe_incl_orig_raydir=False to disable view direction input
        input_dim_viewdir = 3*(gen_cfg.pe_lvl_raydir*2)
        if (gen_cfg.pe_incl_orig_raydir):
            input_dim_viewdir += 3
        print('[Base3DGenerator] Expected viewdir input dimensions: ', input_dim_viewdir)
        self.input_dim_viewdir = input_dim_viewdir

        self.pe_params = [gen_cfg.pe_lvl_feat, gen_cfg.pe_incl_orig_feat,
                          gen_cfg.pe_lvl_raydir, gen_cfg.pe_incl_orig_raydir]

        # Style input dimension
        style_dims = gen_cfg.style_dims
        self.style_dims = style_dims
        interm_style_dims = getattr(gen_cfg, 'interm_style_dims', style_dims)
        self.interm_style_dims = interm_style_dims
        # ---------------------- Style MLP --------------------------
        self.style_net = globals()[gen_cfg.stylenet_model](
            style_dims, interm_style_dims, **gen_cfg.stylenet_model_kwargs)

        # number of output channels for MLP (before blending)
        final_feat_dim = getattr(gen_cfg, 'final_feat_dim', 16)
        self.final_feat_dim = final_feat_dim

        # ----------------------- Sky Network -------------------------
        sky_input_dim_base = 3
        # Dedicated sky network input dimensions
        sky_input_dim = sky_input_dim_base*(gen_cfg.pe_lvl_raydir_sky*2)
        if (gen_cfg.pe_incl_orig_raydir_sky):
            sky_input_dim += sky_input_dim_base
        print('[Base3DGenerator] Expected sky input dimensions: ', sky_input_dim)
        self.pe_params_sky = [gen_cfg.pe_lvl_raydir_sky, gen_cfg.pe_incl_orig_raydir_sky]
        self.sky_net = SKYMLP(sky_input_dim, style_dim=interm_style_dims, out_channels_c=final_feat_dim)

        # ----------------------- Style Encoder -------------------------
        style_enc_cfg = getattr(gen_cfg, 'style_enc', None)
        setattr(style_enc_cfg, 'input_image_channels', 3)
        setattr(style_enc_cfg, 'style_dims', gen_cfg.style_dims)
        self.style_encoder = StyleEncoder(style_enc_cfg)

        # ---------------------- Ray Caster -------------------------
        self.num_blocks_early_stop = gen_cfg.num_blocks_early_stop
        self.num_samples = gen_cfg.num_samples
        self.sample_depth = gen_cfg.sample_depth
        self.coarse_deterministic_sampling = getattr(gen_cfg, 'coarse_deterministic_sampling', True)
        self.sample_use_box_boundaries = getattr(gen_cfg, 'sample_use_box_boundaries', True)

        # ---------------------- Blender -------------------------
        self.raw_noise_std = getattr(gen_cfg, 'raw_noise_std', 0.0)
        self.dists_scale = getattr(gen_cfg, 'dists_scale', 0.25)
        self.clip_feat_map = getattr(gen_cfg, 'clip_feat_map', True)
        self.keep_sky_out = getattr(gen_cfg, 'keep_sky_out', False)
        self.keep_sky_out_avgpool = getattr(gen_cfg, 'keep_sky_out_avgpool', False)
        keep_sky_out_learnbg = getattr(gen_cfg, 'keep_sky_out_learnbg', False)
        self.sky_global_avgpool = getattr(gen_cfg, 'sky_global_avgpool', False)
        if self.keep_sky_out:
            self.sky_replace_color = None
            if keep_sky_out_learnbg:
                sky_replace_color = torch.zeros([final_feat_dim])
                sky_replace_color.requires_grad = True
                self.sky_replace_color = torch.nn.Parameter(sky_replace_color)
        # ---------------------- render_cnn -------------------------
        self.denoiser = RenderCNN(final_feat_dim, style_dim=interm_style_dims)
        self.pad = gen_cfg.pad

    def get_param_groups(self, cfg_opt):
        print('[Generator] get_param_groups')

        if hasattr(cfg_opt, 'ignore_parameters'):
            print('[Generator::get_param_groups] [x]: ignored.')
            optimize_parameters = []
            for k, x in self.named_parameters():
                match = False
                for m in cfg_opt.ignore_parameters:
                    if re.match(m, k) is not None:
                        match = True
                        print(' [x]', k)
                        break
                if match is False:
                    print(' [v]', k)
                    optimize_parameters.append(x)
        else:
            optimize_parameters = self.parameters()

        param_groups = []
        param_groups.append({'params': optimize_parameters})

        if hasattr(cfg_opt, 'param_groups'):
            optimized_param_names = []
            all_param_names = [k for k, v in self.named_parameters()]
            param_groups = []
            for k, v in cfg_opt.param_groups.items():
                print('[Generator::get_param_groups] Adding param group from config:', k, v)
                params = getattr(self, k)
                named_parameters = [k]
                if issubclass(type(params), nn.Module):
                    named_parameters = [k+'.'+pname for pname, _ in params.named_parameters()]
                    params = params.parameters()
                param_groups.append({'params': params, **v})
                optimized_param_names.extend(named_parameters)

        print('[Generator::get_param_groups] UNOPTIMIZED PARAMETERS:\n    ',
              set(all_param_names) - set(optimized_param_names))

        return param_groups

    def _forward_perpix_sub(self, blk_feats, worldcoord2, raydirs_in, z, mc_masks_onehot=None):
        r"""Forwarding the MLP.

        Args:
            blk_feats (K x C1 tensor): Sparse block features.
            worldcoord2 (N x H x W x L x 3 tensor): 3D world coordinates of sampled points.
            raydirs_in (N x H x W x 1 x C2 tensor or None): ray direction embeddings.
            z (N x C3 tensor): Intermediate style vectors.
            mc_masks_onehot (N x H x W x L x C4): One-hot segmentation maps.
        Returns:
            net_out_s (N x H x W x L x 1 tensor): Opacities.
            net_out_c (N x H x W x L x C5 tensor): Color embeddings.
        """
        proj_feature = voxlib.sparse_trilinear_interp_worldcoord(
            blk_feats, self.voxel.corner_t, worldcoord2, ign_zero=True)

        render_net_extra_kwargs = {}
        if self.pe_lvl_localcoords > 0:
            local_coords = torch.remainder(worldcoord2, 1.0) * 2.0
            # Scale to [0, 2], as the positional encoding function doesn't have internal x2
            local_coords[torch.isnan(local_coords)] = 0.0
            local_coords = local_coords.contiguous()
            poscode = voxlib.positional_encoding(local_coords, self.pe_lvl_localcoords, -1, False)
            render_net_extra_kwargs['poscode'] = poscode

        if self.pe_params[0] == 0 and self.pe_params[1] is True:  # no PE shortcut, saves ~400MB
            feature_in = proj_feature
        else:
            if self.pe_no_pe_feat_dim > 0:
                feature_in = voxlib.positional_encoding(
                    proj_feature[..., :-self.pe_no_pe_feat_dim].contiguous(), self.pe_params[0], -1, self.pe_params[1])
                feature_in = torch.cat([feature_in, proj_feature[..., -self.pe_no_pe_feat_dim:]], dim=-1)
            else:
                feature_in = voxlib.positional_encoding(
                    proj_feature.contiguous(), self.pe_params[0], -1, self.pe_params[1])

        net_out_s, net_out_c = self.render_net(feature_in, raydirs_in, z, mc_masks_onehot, **render_net_extra_kwargs)

        if self.raw_noise_std > 0.:
            noise = torch.randn_like(net_out_s) * self.raw_noise_std
            net_out_s = net_out_s + noise

        return net_out_s, net_out_c

    def _forward_perpix(self, blk_feats, voxel_id, depth2, raydirs, cam_ori_t, z):
        r"""Sample points along rays, forwarding the per-point MLP and aggregate pixel features

        Args:
            blk_feats (K x C1 tensor): Sparse block features.
            voxel_id (N x H x W x M x 1 tensor): Voxel ids from ray-voxel intersection test. M: num intersected voxels
            depth2 (N x 2 x H x W x M x 1 tensor): Depths of entrance and exit points for each ray-voxel intersection.
            raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
            cam_ori_t (N x 3 tensor): Camera origins.
            z (N x C3 tensor): Intermediate style vectors.
        """
        # Generate sky_mask; PE transform on ray direction.
        with torch.no_grad():
            raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
            if self.pe_params[2] == 0 and self.pe_params[3] is True:
                raydirs_in = raydirs_in
            elif self.pe_params[2] == 0 and self.pe_params[3] is False:  # Not using raydir at all
                raydirs_in = None
            else:
                raydirs_in = voxlib.positional_encoding(raydirs_in, self.pe_params[2], -1, self.pe_params[3])

            # sky_mask: when True, ray finally hits sky
            sky_mask = voxel_id[:, :, :, [-1], :] == 0
            # sky_only_mask: when True, ray hits nothing but sky
            sky_only_mask = voxel_id[:, :, :, [0], :] == 0

        with torch.no_grad():
            # Random sample points along the ray
            num_samples = self.num_samples + 1
            if self.sample_use_box_boundaries:
                num_samples = self.num_samples - self.num_blocks_early_stop

            # 10 samples per ray + 4 intersections - 2
            rand_depth, new_dists, new_idx = mc_utils.sample_depth_batched(
                depth2, num_samples, deterministic=self.coarse_deterministic_sampling,
                use_box_boundaries=self.sample_use_box_boundaries, sample_depth=self.sample_depth)

            worldcoord2 = raydirs * rand_depth + cam_ori_t[:, None, None, None, :]

            # Generate per-sample segmentation label
            voxel_id_reduced = self.label_trans.mc2reduced(voxel_id, ign2dirt=True)
            mc_masks = torch.gather(voxel_id_reduced, -2, new_idx)  # B 256 256 N 1
            mc_masks = mc_masks.long()
            mc_masks_onehot = torch.zeros([mc_masks.size(0), mc_masks.size(1), mc_masks.size(
                2), mc_masks.size(3), self.num_reduced_labels], dtype=torch.float, device=voxel_id.device)
            # mc_masks_onehot: [B H W Nlayer 680]
            mc_masks_onehot.scatter_(-1, mc_masks, 1.0)

        net_out_s, net_out_c = self._forward_perpix_sub(blk_feats, worldcoord2, raydirs_in, z, mc_masks_onehot)

        # Handle sky
        sky_raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
        sky_raydirs_in = voxlib.positional_encoding(sky_raydirs_in, self.pe_params_sky[0], -1, self.pe_params_sky[1])
        skynet_out_c = self.sky_net(sky_raydirs_in, z)

        # Blending
        weights = mc_utils.volum_rendering_relu(net_out_s, new_dists * self.dists_scale, dim=-2)

        # If a ray exclusively hits the sky (no intersection with the voxels), set its weight to zero.
        weights = weights * torch.logical_not(sky_only_mask).float()
        total_weights_raw = torch.sum(weights, dim=-2, keepdim=True)  # 256 256 1 1
        total_weights = total_weights_raw

        is_gnd = worldcoord2[..., [0]] <= 1.0  # Y X Z, [256, 256, 4, 3], nan < 1.0 == False
        is_gnd = is_gnd.any(dim=-2, keepdim=True)
        nosky_mask = torch.logical_or(torch.logical_not(sky_mask), is_gnd)
        nosky_mask = nosky_mask.float()

        # Avoid sky leakage
        sky_weight = 1.0-total_weights
        if self.keep_sky_out:
            # keep_sky_out_avgpool overrides sky_replace_color
            if self.sky_replace_color is None or self.keep_sky_out_avgpool:
                if self.keep_sky_out_avgpool:
                    if hasattr(self, 'sky_avg'):
                        sky_avg = self.sky_avg
                    else:
                        if self.sky_global_avgpool:
                            sky_avg = torch.mean(skynet_out_c, dim=[1, 2], keepdim=True)
                        else:
                            skynet_out_c_nchw = skynet_out_c.permute(0, 4, 1, 2, 3).squeeze(-1)
                            sky_avg = F.avg_pool2d(skynet_out_c_nchw, 31, stride=1, padding=15, count_include_pad=False)
                            sky_avg = sky_avg.permute(0, 2, 3, 1).unsqueeze(-2)
                    # print(sky_avg.shape)
                    skynet_out_c = skynet_out_c * (1.0-nosky_mask) + sky_avg*(nosky_mask)
                else:
                    sky_weight = sky_weight * (1.0-nosky_mask)
            else:
                skynet_out_c = skynet_out_c * (1.0-nosky_mask) + self.sky_replace_color*(nosky_mask)

        if self.clip_feat_map is True:  # intermediate feature before blending & CNN
            rgbs = torch.clamp(net_out_c, -1, 1) + 1
            rgbs_sky = torch.clamp(skynet_out_c, -1, 1) + 1
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
            net_out = net_out - 1
        elif self.clip_feat_map is False:
            rgbs = net_out_c
            rgbs_sky = skynet_out_c
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
        elif self.clip_feat_map == 'tanh':
            rgbs = torch.tanh(net_out_c)
            rgbs_sky = torch.tanh(skynet_out_c)
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
        else:
            raise NotImplementedError

        return net_out, new_dists, weights, total_weights_raw, rand_depth, net_out_s, net_out_c, skynet_out_c, \
            nosky_mask, sky_mask, sky_only_mask, new_idx

    def _forward_global(self, net_out, z):
        r"""Forward the CNN

        Args:
            net_out (N x C5 x H x W tensor): Intermediate feature maps.
            z (N x C3 tensor): Intermediate style vectors.

        Returns:
            fake_images (N x 3 x H x W tensor): Output image.
            fake_images_raw (N x 3 x H x W tensor): Output image before TanH.
        """
        fake_images = net_out.permute(0, 3, 1, 2)
        fake_images_raw = self.denoiser(fake_images, z)
        fake_images = torch.tanh(fake_images_raw)

        return fake_images, fake_images_raw
