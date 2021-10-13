# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import imaginaire.model_utils.gancraft.camctl as camctl
import imaginaire.model_utils.gancraft.mc_utils as mc_utils
import imaginaire.model_utils.gancraft.voxlib as voxlib
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.generators.gancraft_base import Base3DGenerator, RenderMLP  # noqa


class Generator(Base3DGenerator):
    r"""GANcraft generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super(Generator, self).__init__(gen_cfg, data_cfg)
        print('GANcraft generator initialization.')

        # Load voxels of the input world.
        # The loaded voxel tensor has a shape of [X, Y, Z], dtype==torch.int32
        # 0 means empty (air).
        print('[Generator] Loading voxel world: ', gen_cfg.voxel_path)
        if gen_cfg.voxel_path.endswith('.npy'):
            voxel_t = np.load(gen_cfg.voxel_path)
            voxel_t = torch.from_numpy(voxel_t.astype(np.int32))
        else:
            voxel_t = mc_utils.load_voxel_new(gen_cfg.voxel_path, shape=gen_cfg.voxel_shape)
        print('[Generator] Loaded voxel world.')
        self.voxel = mc_utils.McVoxel(voxel_t, preproc_ver=gen_cfg.voxel_preproc_ver)
        blk_feats = torch.empty([self.voxel.nfilledvox, gen_cfg.blk_feat_dim], requires_grad=True)
        self.blk_feats = nn.Parameter(blk_feats)  # Feature per voxel corner.

        # Minecraft -> SPADE label translator.
        self.label_trans = mc_utils.MCLabelTranslator()
        self.num_reduced_labels = self.label_trans.get_num_reduced_lbls()
        self.reduced_label_set = getattr(gen_cfg, 'reduced_label_set', False)
        self.use_label_smooth = getattr(gen_cfg, 'use_label_smooth', False)
        self.use_label_smooth_real = getattr(gen_cfg, 'use_label_smooth_real', self.use_label_smooth)
        self.use_label_smooth_pgt = getattr(gen_cfg, 'use_label_smooth_pgt', False)
        self.label_smooth_dia = getattr(gen_cfg, 'label_smooth_dia', 11)

        # Load MLP model.
        self.render_net = globals()[gen_cfg.mlp_model](
            self.input_dim, viewdir_dim=self.input_dim_viewdir, style_dim=self.interm_style_dims,
            mask_dim=self.num_reduced_labels, out_channels_s=1, out_channels_c=self.final_feat_dim,
            **self.mlp_model_kwargs)

        # Camera sampler.
        self.camera_sampler_type = getattr(gen_cfg, 'camera_sampler_type', "random")
        assert self.camera_sampler_type in ['random', 'traditional']
        self.camera_min_entropy = getattr(gen_cfg, 'camera_min_entropy', -1)
        self.camera_rej_avg_depth = getattr(gen_cfg, 'camera_rej_avg_depth', -1)
        self.cam_res = gen_cfg.cam_res
        self.crop_size = gen_cfg.crop_size

        print('Done with the GANcraft generator initialization.')

    def custom_init(self):
        r"""Weight initialization of GANcraft components."""

        self.blk_feats.data.uniform_(-1, 1)

        def init_func(m):
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight.data, a=0.2, nonlinearity='leaky_relu')
                m.weight.data *= 0.5
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(0.0)
        self.apply(init_func)

    def _get_batch(self, batch_size, device):
        r"""Sample camera poses and perform ray-voxel intersection.

        Args:
            batch_size (int): Expected batch size of the current batch
            device (torch.device): Device on which the tensors should be stored
        """
        with torch.no_grad():
            voxel_id_batch = []
            depth2_batch = []
            raydirs_batch = []
            cam_ori_t_batch = []
            for b in range(batch_size):
                while True:  # Rejection sampling.
                    # Sample camera pose.
                    if self.camera_sampler_type == 'random':
                        cam_res = self.cam_res
                        cam_ori_t, cam_dir_t, cam_up_t = camctl.rand_camera_pose_thridperson2(self.voxel)
                        # ~24mm fov horizontal.
                        cam_f = 0.5/np.tan(np.deg2rad(73/2) * (np.random.rand(1)*0.5+0.5)) * (cam_res[1]-1)
                        cam_c = [(cam_res[0]-1)/2, (cam_res[1]-1)/2]
                        cam_res_crop = [self.crop_size[0] + self.pad, self.crop_size[1] + self.pad]
                        cam_c = mc_utils.rand_crop(cam_c, cam_res, cam_res_crop)
                    elif self.camera_sampler_type == 'traditional':
                        cam_res = self.cam_res
                        cam_c = [(cam_res[0]-1)/2, (cam_res[1]-1)/2]
                        dice = torch.rand(1).item()
                        if dice > 0.5:
                            cam_ori_t, cam_dir_t, cam_up_t, cam_f = \
                                camctl.rand_camera_pose_tour(self.voxel)
                            cam_f = cam_f * (cam_res[1]-1)
                        else:
                            cam_ori_t, cam_dir_t, cam_up_t = \
                                camctl.rand_camera_pose_thridperson2(self.voxel)
                            # ~24mm fov horizontal.
                            cam_f = 0.5 / np.tan(np.deg2rad(73/2) * (np.random.rand(1)*0.5+0.5)) * (cam_res[1]-1)

                        cam_res_crop = [self.crop_size[0] + self.pad, self.crop_size[1] + self.pad]
                        cam_c = mc_utils.rand_crop(cam_c, cam_res, cam_res_crop)
                    else:
                        raise NotImplementedError(
                            'Unknown self.camera_sampler_type: {}'.format(self.camera_sampler_type))
                    # Run ray-voxel intersection test
                    r"""Ray-voxel intersection CUDA kernel.
                    Note: voxel_id = 0 and depth2 = NaN if there is no intersection along the ray

                    Args:
                        voxel_t (Y x 512 x 512 tensor, int32): Full 3D voxel of MC block IDs.
                        cam_ori_t (3 tensor): Camera origin.
                        cam_dir_t (3 tensor): Camera direction.
                        cam_up_t (3 tensor): Camera up vector.
                        cam_f (float): Camera focal length (in pixels).
                        cam_c  (list of 2 floats [x, y]): Camera optical center.
                        img_dims (list of 2 ints [H, W]): Camera resolution.
                        max_samples (int): Maximum number of blocks intersected along the ray before stopping.
                    Returns:
                        voxel_id (    img_dims[0] x img_dims[1] x max_samples x 1 tensor): IDs of intersected tensors
                        along each ray
                        depth2   (2 x img_dims[0] x img_dims[1] x max_samples x 1 tensor): Depths of entrance and exit
                        points for each ray-voxel intersection.
                        raydirs  (    img_dims[0] x img_dims[1] x 1 x 3 tensor): The direction of each ray.

                    """
                    voxel_id, depth2, raydirs = voxlib.ray_voxel_intersection_perspective(
                        self.voxel.voxel_t, cam_ori_t, cam_dir_t, cam_up_t, cam_f, cam_c, cam_res_crop,
                        self.num_blocks_early_stop)

                    if self.camera_rej_avg_depth > 0:
                        depth_map = depth2[0, :, :, 0, :]
                        avg_depth = torch.mean(depth_map[~torch.isnan(depth_map)])
                        if avg_depth < self.camera_rej_avg_depth:
                            continue

                    # Reject low entropy.
                    if self.camera_min_entropy > 0:
                        # Check entropy.
                        maskcnt = torch.bincount(
                            torch.flatten(voxel_id[:, :, 0, 0]), weights=None, minlength=680).float() / \
                            (voxel_id.size(0)*voxel_id.size(1))
                        maskentropy = -torch.sum(maskcnt * torch.log(maskcnt+1e-10))
                        if maskentropy < self.camera_min_entropy:
                            continue
                    break

                voxel_id_batch.append(voxel_id)
                depth2_batch.append(depth2)
                raydirs_batch.append(raydirs)
                cam_ori_t_batch.append(cam_ori_t)
            voxel_id = torch.stack(voxel_id_batch, dim=0)
            depth2 = torch.stack(depth2_batch, dim=0)
            raydirs = torch.stack(raydirs_batch, dim=0)
            cam_ori_t = torch.stack(cam_ori_t_batch, dim=0).to(device)
            cam_poses = None
        return voxel_id, depth2, raydirs, cam_ori_t, cam_poses

    def get_pseudo_gt(self, pseudo_gen, voxel_id, z=None, style_img=None, resize_512=True, deterministic=False):
        r"""Evaluating img2img network to obtain pseudo-ground truth images.

        Args:
            pseudo_gen (callable): Function converting mask to image using img2img network.
            voxel_id (N x img_dims[0] x img_dims[1] x max_samples x 1 tensor): IDs of intersected tensors along
            each ray.
            z (N x C tensor): Optional style code passed to pseudo_gen.
            style_img (N x 3 x H x W tensor): Optional style image passed to pseudo_gen.
            resize_512 (bool): If True, evaluate pseudo_gen at 512x512 regardless of input resolution.
            deterministic (bool): If True, disable stochastic label mapping.
        """
        with torch.no_grad():
            mc_mask = voxel_id[:, :, :, 0, :].permute(0, 3, 1, 2).long()
            coco_mask = self.label_trans.mc2coco(mc_mask) - 1
            coco_mask[coco_mask < 0] = 183

            if not deterministic:
                # Stochastic mapping
                dice = torch.rand(1).item()
                if dice > 0.5 and dice < 0.9:
                    coco_mask[coco_mask == self.label_trans.gglbl2ggid('sky')] = self.label_trans.gglbl2ggid('clouds')
                elif dice >= 0.9:
                    coco_mask[coco_mask == self.label_trans.gglbl2ggid('sky')] = self.label_trans.gglbl2ggid('fog')
                dice = torch.rand(1).item()
                if dice > 0.33 and dice < 0.66:
                    coco_mask[coco_mask == self.label_trans.gglbl2ggid('water')] = self.label_trans.gglbl2ggid('sea')
                elif dice >= 0.66:
                    coco_mask[coco_mask == self.label_trans.gglbl2ggid('water')] = self.label_trans.gglbl2ggid('river')

            fake_masks = torch.zeros([coco_mask.size(0), 185, coco_mask.size(2), coco_mask.size(3)],
                                     dtype=torch.half, device=voxel_id.device)
            fake_masks.scatter_(1, coco_mask, 1.0)

            if self.use_label_smooth_pgt:
                fake_masks = mc_utils.segmask_smooth(fake_masks, kernel_size=self.label_smooth_dia)
            if self.pad > 0:
                fake_masks = fake_masks[:, :, self.pad//2:-self.pad//2, self.pad//2:-self.pad//2]

            # Generate pseudo GT using GauGAN.
            if resize_512:
                fake_masks_512 = F.interpolate(fake_masks, size=[512, 512], mode='nearest')
            else:
                fake_masks_512 = fake_masks
            pseudo_real_img = pseudo_gen(fake_masks_512, z=z, style_img=style_img)

            # NaN Inf Guard. NaN can occure on Volta GPUs.
            nan_mask = torch.isnan(pseudo_real_img)
            inf_mask = torch.isinf(pseudo_real_img)
            pseudo_real_img[nan_mask | inf_mask] = 0.0
            if resize_512:
                pseudo_real_img = F.interpolate(
                    pseudo_real_img, size=[fake_masks.size(2), fake_masks.size(3)], mode='area')
            pseudo_real_img = torch.clamp(pseudo_real_img, -1, 1)

        return pseudo_real_img, fake_masks

    def sample_camera(self, data, pseudo_gen):
        r"""Sample camera randomly and precompute everything used by both Gen and Dis.

        Args:
            data (dict):
                images (N x 3 x H x W tensor) : Real images
                label (N x C2 x H x W tensor) : Segmentation map
            pseudo_gen (callable): Function converting mask to image using img2img network.
        Returns:
            ret (dict):
                voxel_id (N x H x W x max_samples x 1 tensor): IDs of intersected tensors along each ray.
                depth2 (N x 2 x H x W x max_samples x 1 tensor): Depths of entrance and exit points for each ray-voxel
                intersection.
                raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
                cam_ori_t (N x 3 tensor): Camera origins.
                pseudo_real_img (N x 3 x H x W tensor): Pseudo-ground truth image.
                real_masks (N x C3 x H x W tensor): One-hot segmentation map for real images, with translated labels.
                fake_masks (N x C3 x H x W tensor): One-hot segmentation map for sampled camera views.
        """
        device = torch.device('cuda')
        batch_size = data['images'].size(0)
        # ================ Assemble a batch ==================
        # Requires: voxel_id, depth2, raydirs, cam_ori_t.
        voxel_id, depth2, raydirs, cam_ori_t, _ = self._get_batch(batch_size, device)
        ret = {'voxel_id': voxel_id, 'depth2': depth2, 'raydirs': raydirs, 'cam_ori_t': cam_ori_t}

        if pseudo_gen is not None:
            pseudo_real_img, _ = self.get_pseudo_gt(pseudo_gen, voxel_id)
        ret['pseudo_real_img'] = pseudo_real_img.float()

        # =============== Mask translation ================
        real_masks = data['label']
        if self.reduced_label_set:
            # Translate fake mask (directly from mcid).
            # convert unrecognized labels to 'dirt'.
            # N C H W [1 1 80 80]
            reduce_fake_mask = self.label_trans.mc2reduced(
                voxel_id[:, :, :, 0, :].permute(0, 3, 1, 2).long(), ign2dirt=True)
            reduce_fake_mask_onehot = torch.zeros([
                reduce_fake_mask.size(0), self.num_reduced_labels, reduce_fake_mask.size(2), reduce_fake_mask.size(3)],
                dtype=torch.float, device=device)
            reduce_fake_mask_onehot.scatter_(1, reduce_fake_mask, 1.0)
            fake_masks = reduce_fake_mask_onehot
            if self.pad != 0:
                fake_masks = fake_masks[:, :, self.pad//2:-self.pad//2, self.pad//2:-self.pad//2]

            # Translate real mask (data['label']), which is onehot.
            real_masks_idx = torch.argmax(real_masks, dim=1, keepdim=True)
            real_masks_idx[real_masks_idx > 182] = 182

            reduced_real_mask = self.label_trans.coco2reduced(real_masks_idx)
            reduced_real_mask_onehot = torch.zeros([
                reduced_real_mask.size(0), self.num_reduced_labels, reduced_real_mask.size(2),
                reduced_real_mask.size(3)], dtype=torch.float, device=device)
            reduced_real_mask_onehot.scatter_(1, reduced_real_mask, 1.0)
            real_masks = reduced_real_mask_onehot

        # Mask smoothing.
        if self.use_label_smooth:
            fake_masks = mc_utils.segmask_smooth(fake_masks, kernel_size=self.label_smooth_dia)
        if self.use_label_smooth_real:
            real_masks = mc_utils.segmask_smooth(real_masks, kernel_size=self.label_smooth_dia)

        ret['real_masks'] = real_masks
        ret['fake_masks'] = fake_masks

        return ret

    def forward(self, data, random_style=False):
        r"""GANcraft Generator forward.

        Args:
            data (dict):
                images (N x 3 x H x W tensor) : Real images
                voxel_id (N x H x W x max_samples x 1 tensor): IDs of intersected tensors along each ray.
                depth2 (N x 2 x H x W x max_samples x 1 tensor): Depths of entrance and exit points for each ray-voxel
                intersection.
                raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
                cam_ori_t (N x 3 tensor): Camera origins.
            random_style (bool): Whether to sample a random style vector.
        Returns:
            output (dict):
                fake_images (N x 3 x H x W tensor): fake images
                mu (N x C1 tensor): mean vectors
                logvar (N x C1 tensor): log-variance vectors
        """
        device = torch.device('cuda')
        batch_size = data['images'].size(0)

        # ================ Assemble a batch ==================
        # Requires: voxel_id, depth2, raydirs, cam_ori_t.
        voxel_id, depth2, raydirs, cam_ori_t = data['voxel_id'], data['depth2'], data['raydirs'], data['cam_ori_t']
        if 'pseudo_real_img' in data:
            pseudo_real_img = data['pseudo_real_img']

        z, mu, logvar = None, None, None
        if random_style:
            if self.style_dims > 0:
                z = torch.randn(batch_size, self.style_dims, dtype=torch.float32, device=device)
        else:
            if self.style_encoder is None:
                # ================ Get Style Code =================
                if self.style_dims > 0:
                    z = torch.randn(batch_size, self.style_dims, dtype=torch.float32, device=device)
            else:
                mu, logvar, z = self.style_encoder(pseudo_real_img)

        # ================ Network Forward ================
        # Forward StyleNet
        if self.style_net is not None:
            z = self.style_net(z)

        # Forward per-pixel net.
        net_out, new_dists, weights, total_weights_raw, rand_depth, net_out_s, net_out_c, skynet_out_c, nosky_mask, \
            sky_mask, sky_only_mask, new_idx = self._forward_perpix(
                self.blk_feats, voxel_id, depth2, raydirs, cam_ori_t, z)

        # Forward global net.
        fake_images, fake_images_raw = self._forward_global(net_out, z)
        if self.pad != 0:
            fake_images = fake_images[:, :, self.pad//2:-self.pad//2, self.pad//2:-self.pad//2]

        # =============== Arrange Return Values ================
        output = {}
        output['fake_images'] = fake_images
        output['mu'] = mu
        output['logvar'] = logvar
        return output

    def inference(self,
                  output_dir,
                  camera_mode,
                  style_img_path=None,
                  seed=1,
                  pad=30,
                  num_samples=40,
                  num_blocks_early_stop=6,
                  sample_depth=3,
                  tile_size=128,
                  resolution_hw=[540, 960],
                  cam_ang=72,
                  cam_maxstep=10):
        r"""Compute result images according to the provided camera trajectory and save the results in the specified
        folder. The full image is evaluated in multiple tiles to save memory.

        Args:
            output_dir (str): Where should the results be stored.
            camera_mode (int): Which camera trajectory to use.
            style_img_path (str): Path to the style-conditioning image.
            seed (int): Random seed (controls style when style_image_path is not specified).
            pad (int): Pixels to remove from the image tiles before stitching. Should be equal or larger than the
            receptive field of the CNN to avoid border artifact.
            num_samples (int): Number of samples per ray (different from training).
            num_blocks_early_stop (int): Max number of intersected boxes per ray before stopping
            (different from training).
            sample_depth (float): Max distance traveled through boxes before stopping (different from training).
            tile_size (int): Max size of a tile in pixels.
            resolution_hw (list [H, W]): Resolution of the output image.
            cam_ang (float): Horizontal FOV of the camera (may be adjusted by the camera controller).
            cam_maxstep (int): Number of frames sampled from the camera trajectory.
        """

        def write_img(path, img, rgb_input=False):
            img = ((img*0.5+0.5)*255).detach().cpu().numpy().astype(np.uint8)
            img = img[0].transpose(1, 2, 0)
            if rgb_input:
                img = img[..., [2, 1, 0]]
            cv2.imwrite(path, img,  [cv2.IMWRITE_PNG_COMPRESSION, 4])
            return img[..., ::-1]

        def read_img(path):
            img = cv2.imread(path).astype(np.float32)[..., [2, 1, 0]].transpose(2, 0, 1) / 255
            img = img * 2 - 1
            img = torch.from_numpy(img)

        print('Saving to', output_dir)

        # Use provided random seed.
        device = torch.device('cuda')
        rng_cuda = torch.Generator(device=device)
        rng_cuda = rng_cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.pad = pad
        self.num_samples = num_samples
        self.num_blocks_early_stop = num_blocks_early_stop
        self.sample_depth = sample_depth

        self.coarse_deterministic_sampling = True
        self.crop_size = resolution_hw
        self.cam_res = [self.crop_size[0]+self.pad, self.crop_size[1]+self.pad]
        self.use_label_smooth_pgt = False

        # Make output dirs.
        gancraft_outputs_dir = os.path.join(output_dir, 'gancraft_outputs')
        os.makedirs(gancraft_outputs_dir, exist_ok=True)
        vis_masks_dir = os.path.join(output_dir, 'vis_masks')
        os.makedirs(vis_masks_dir, exist_ok=True)
        fout = imageio.get_writer(gancraft_outputs_dir + '.mp4', fps=1)
        fout_cat = imageio.get_writer(gancraft_outputs_dir + '-vis_masks.mp4', fps=1)

        evalcamctl = camctl.EvalCameraController(
            self.voxel, maxstep=cam_maxstep, pattern=camera_mode, cam_ang=cam_ang,
            smooth_decay_multiplier=150/cam_maxstep)

        # Get output style.
        if style_img_path is None:
            z = torch.empty(1, self.style_dims, dtype=torch.float32, device=device)
            z.normal_(generator=rng_cuda)
        else:
            style_img = read_img(style_img_path)
            style_img = style_img.to(device).unsqueeze(0)
            mu, logvar, z = self.style_encoder(style_img)
        z = self.style_net(z)

        # Generate required output images.
        for id, (cam_ori_t, cam_dir_t, cam_up_t, cam_f) in enumerate(evalcamctl):
            print('Rendering frame', id)
            cam_f = cam_f * (self.crop_size[1]-1)  # So that the view is not depending on the padding
            cam_c = [(self.cam_res[0]-1)/2, (self.cam_res[1]-1)/2]

            voxel_id, depth2, raydirs = voxlib.ray_voxel_intersection_perspective(
                self.voxel.voxel_t, cam_ori_t, cam_dir_t, cam_up_t, cam_f, cam_c, self.cam_res,
                self.num_blocks_early_stop)

            voxel_id = voxel_id.unsqueeze(0)
            depth2 = depth2.unsqueeze(0)
            raydirs = raydirs.unsqueeze(0)
            cam_ori_t = cam_ori_t.unsqueeze(0).to(device)

            # Save 3D voxel rendering.
            mc_rgb = self.label_trans.mc_color(voxel_id[0, :, :, 0, 0].cpu().numpy())
            # Diffused shading, co-located light.
            first_intersection_depth = depth2[:, 0, :, :, 0, None, :]  # [1, 542, 542, 1, 1].
            first_intersection_point = raydirs * first_intersection_depth + cam_ori_t[:, None, None, None, :]
            fip_local_coords = torch.remainder(first_intersection_point, 1.0)
            fip_wall_proximity = torch.minimum(fip_local_coords, 1.0-fip_local_coords)
            fip_wall_orientation = torch.argmin(fip_wall_proximity, dim=-1, keepdim=False)
            # 0: [1,0,0]; 1: [0,1,0]; 2: [0,0,1]
            lut = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32,
                               device=fip_wall_orientation.device)
            fip_normal = lut[fip_wall_orientation]  # [1, 542, 542, 1, 3]
            diffuse_shade = torch.abs(torch.sum(fip_normal * raydirs, dim=-1))

            mc_rgb = (mc_rgb.astype(np.float) / 255) ** 2.2
            mc_rgb = mc_rgb * diffuse_shade[0, :, :, :].cpu().numpy()
            mc_rgb = (mc_rgb ** (1/2.2)) * 255
            mc_rgb = mc_rgb.astype(np.uint8)
            if self.pad > 0:
                mc_rgb = mc_rgb[self.pad//2:-self.pad//2, self.pad//2:-self.pad//2]
            cv2.imwrite(os.path.join(vis_masks_dir, '{:05d}.png'.format(id)), mc_rgb,  [cv2.IMWRITE_PNG_COMPRESSION, 4])

            # Tiled eval of GANcraft.
            voxel_id_all = voxel_id
            depth2_all = depth2
            raydirs_all = raydirs

            # Evaluate sky in advance to get a consistent sky in the semi-transparent region.
            if self.sky_global_avgpool:
                sky_raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
                sky_raydirs_in = voxlib.positional_encoding(
                    sky_raydirs_in, self.pe_params_sky[0], -1, self.pe_params_sky[1])
                skynet_out_c = self.sky_net(sky_raydirs_in, z)
                sky_avg = torch.mean(skynet_out_c, dim=[1, 2], keepdim=True)
                self.sky_avg = sky_avg

            num_strips_h = (self.cam_res[0]-self.pad+tile_size-1)//tile_size
            num_strips_w = (self.cam_res[1]-self.pad+tile_size-1)//tile_size

            fake_images_chunks_v = []
            # For each horizontal strip.
            for strip_id_h in range(num_strips_h):
                strip_begin_h = strip_id_h * tile_size
                strip_end_h = np.minimum(strip_id_h * tile_size + tile_size + self.pad, self.cam_res[0])
                # For each vertical strip.
                fake_images_chunks_h = []
                for strip_id_w in range(num_strips_w):
                    strip_begin_w = strip_id_w * tile_size
                    strip_end_w = np.minimum(strip_id_w * tile_size + tile_size + self.pad, self.cam_res[1])

                    voxel_id = voxel_id_all[:, strip_begin_h:strip_end_h, strip_begin_w:strip_end_w, :, :]
                    depth2 = depth2_all[:, :, strip_begin_h:strip_end_h, strip_begin_w:strip_end_w, :, :]
                    raydirs = raydirs_all[:, strip_begin_h:strip_end_h, strip_begin_w:strip_end_w, :, :]

                    net_out, new_dists, weights, total_weights_raw, rand_depth, net_out_s, net_out_c, skynet_out_c, \
                        nosky_mask, sky_mask, sky_only_mask, new_idx = self._forward_perpix(
                            self.blk_feats, voxel_id, depth2, raydirs, cam_ori_t, z)
                    fake_images, _ = self._forward_global(net_out, z)

                    if self.pad != 0:
                        fake_images = fake_images[:, :, self.pad//2:-self.pad//2, self.pad//2:-self.pad//2]
                    fake_images_chunks_h.append(fake_images)
                fake_images_h = torch.cat(fake_images_chunks_h, dim=-1)
                fake_images_chunks_v.append(fake_images_h)
            fake_images = torch.cat(fake_images_chunks_v, dim=-2)
            rgb = write_img(os.path.join(gancraft_outputs_dir,
                            '{:05d}.png'.format(id)), fake_images, rgb_input=True)
            fout.append_data(rgb)
            fout_cat.append_data(np.concatenate((mc_rgb[..., ::-1], rgb), axis=1))
        fout.close()
        fout_cat.close()
