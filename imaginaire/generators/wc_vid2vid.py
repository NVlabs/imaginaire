# share: outside-ok
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from imaginaire.config import Config
from imaginaire.generators.vid2vid import Generator as Vid2VidGenerator
from imaginaire.model_utils.fs_vid2vid import resample
from imaginaire.model_utils.wc_vid2vid.render import SplatRenderer
from imaginaire.utils.trainer import (get_model_optimizer_and_scheduler,
                                      get_trainer)
from imaginaire.utils.visualization import tensor2im


class Generator(Vid2VidGenerator):
    r"""world consistent vid2vid generator constructor.

    Args:
       gen_cfg (obj): Generator definition part of the yaml config file.
       data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, gen_cfg, data_cfg):
        # Guidance options.
        self.guidance_cfg = gen_cfg.guidance
        self.guidance_only_with_flow = getattr(
            self.guidance_cfg, 'only_with_flow', False)
        self.guidance_partial_conv = getattr(
            self.guidance_cfg, 'partial_conv', False)

        # Splatter for guidance.
        self.renderer = SplatRenderer()
        self.reset_renderer()

        # Single image model.
        self.single_image_model = None

        # Initialize the rest same as vid2vid.
        super().__init__(gen_cfg, data_cfg)

    def _init_single_image_model(self, load_weights=True):
        r"""Load single image model, if any."""
        if self.single_image_model is None and \
                hasattr(self.gen_cfg, 'single_image_model'):
            print('Using single image model...')
            single_image_cfg = Config(self.gen_cfg.single_image_model.config)

            # Init model.
            net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
                get_model_optimizer_and_scheduler(single_image_cfg)

            # Init trainer and load checkpoint.
            trainer = get_trainer(single_image_cfg, net_G, net_D,
                                  opt_G, opt_D,
                                  sch_G, sch_D,
                                  None, None)
            if load_weights:
                print('Loading single image model checkpoint')
                single_image_ckpt = self.gen_cfg.single_image_model.checkpoint
                trainer.load_checkpoint(single_image_cfg, single_image_ckpt)
                print('Loaded single image model checkpoint')

            self.single_image_model = net_G.module
            self.single_image_model_z = None

    def reset_renderer(self, is_flipped_input=False):
        r"""Reset the renderer.
        Args:
            is_flipped_input (bool): Is the input sequence left-right flipped?
        """
        self.renderer.reset()
        self.is_flipped_input = is_flipped_input
        self.renderer_num_forwards = 0
        self.single_image_model_z = None

    def renderer_update_point_cloud(self, image, point_info):
        r"""Update the renderer's color dictionary."""
        if point_info is None or len(point_info) == 0:
            return
        # print('Updating the renderer.')
        _, _, h, w = image.size()

        # Renderer expects (h, w, c) [0-255] RGB image.
        if isinstance(image, torch.Tensor):
            image = tensor2im(image.detach())[0]

        # Flip this image to correspond to SfM camera pose.
        if self.is_flipped_input:
            image = np.fliplr(image).copy()

        self.renderer.update_point_cloud(image, point_info)
        self.renderer_num_forwards += 1

    def get_guidance_images_and_masks(self, unprojection):
        r"""Do stuff."""

        resolution = 'w1024xh512'
        point_info = unprojection[resolution]

        w, h = resolution.split('x')
        w, h = int(w[1:]), int(h[1:])

        # This returns guidance image in [0-255] RGB.
        # We will convert it into Tensor repr. below.
        guidance_image, guidance_mask = self.renderer.render_image(
            point_info, w, h, return_mask=True)

        # If mask is None, there is no guidance.
        # print(np.sum(guidance_mask), guidance_mask.size)
        # if np.sum(guidance_mask) == 0:
        #     return None, point_info

        # Flip guidance image and guidance mask if needed.
        if self.is_flipped_input:
            guidance_image = np.fliplr(guidance_image).copy()
            guidance_mask = np.fliplr(guidance_mask).copy()

        # Go from (h, w, c) to (1, c, h, w).
        # Convert guidance image to Tensor.
        guidance_image = (transforms.ToTensor()(guidance_image) - 0.5) * 2
        guidance_mask = transforms.ToTensor()(guidance_mask)
        guidance = torch.cat((guidance_image, guidance_mask), dim=0)
        guidance = guidance.unsqueeze(0).cuda()

        # Save guidance at all resolutions.
        guidance_images_and_masks = guidance

        return guidance_images_and_masks, point_info

    def forward(self, data):
        r"""vid2vid generator forward.
        Args:
           data (dict) : Dictionary of input data.
        Returns:
           output (dict) : Dictionary of output data.
        """
        self._init_single_image_model()

        label = data['label']
        unprojection = data['unprojection']
        label_prev, img_prev = data['prev_labels'], data['prev_images']
        is_first_frame = img_prev is None
        z = getattr(data, 'z', None)
        bs, _, h, w = label.size()

        # Whether to warp the previous frame or not.
        flow = mask = img_warp = None
        warp_prev = self.temporal_initialized and not is_first_frame and \
            label_prev.shape[1] == self.num_frames_G - 1

        # Get guidance images and masks.
        guidance_images_and_masks, point_info = None, None
        if unprojection is not None:
            guidance_images_and_masks, point_info = \
                self.get_guidance_images_and_masks(unprojection)

        # Get SPADE conditional maps by embedding current label input.
        cond_maps_now = self.get_cond_maps(label, self.label_embedding)

        # Use single image model, if flow features are not available.
        # Guidance features are used whenever flow features are available.
        if self.single_image_model is not None and not warp_prev:
            # Get z vector for single image model.
            if self.single_image_model_z is None:
                bs = data['label'].size(0)
                z = torch.randn(bs, self.single_image_model.style_dims,
                                dtype=torch.float32).cuda()
                if data['label'].dtype == torch.float16:
                    z = z.half()
                self.single_image_model_z = z

            # Get output image.
            data['z'] = self.single_image_model_z
            self.single_image_model.eval()
            with torch.no_grad():
                output = self.single_image_model.spade_generator(data)
            img_final = output['fake_images'].detach()
            fake_images_source = 'pretrained'
        else:
            # Input to the generator will either be noise/segmentation map (for
            # first frame) or encoded previous frame (for subsequent frames).
            if is_first_frame:
                # First frame in the sequence, start from scratch.
                if self.use_segmap_as_input:
                    x_img = F.interpolate(label, size=(self.sh, self.sw))
                    x_img = self.fc(x_img)
                else:
                    if z is None:
                        z = torch.randn(bs, self.z_dim, dtype=label.dtype,
                                        device=label.get_device()).fill_(0)
                    x_img = self.fc(z).view(bs, -1, self.sh, self.sw)

                # Upsampling layers.
                for i in range(self.num_layers, self.num_downsamples_img, -1):
                    j = min(self.num_downsamples_embed, i)
                    x_img = getattr(self, 'up_' + str(i)
                                    )(x_img, *cond_maps_now[j])
                    x_img = self.upsample(x_img)
            else:
                # Not the first frame, will encode the previous frame and feed
                # to the generator.
                x_img = self.down_first(img_prev[:, -1])

                # Get label embedding for the previous frame.
                cond_maps_prev = self.get_cond_maps(label_prev[:, -1],
                                                    self.label_embedding)

                # Downsampling layers.
                for i in range(self.num_downsamples_img + 1):
                    j = min(self.num_downsamples_embed, i)
                    x_img = getattr(self, 'down_' + str(i))(x_img,
                                                            *cond_maps_prev[j])
                    if i != self.num_downsamples_img:
                        x_img = self.downsample(x_img)

                # Resnet blocks.
                j = min(self.num_downsamples_embed,
                        self.num_downsamples_img + 1)
                for i in range(self.num_res_blocks):
                    cond_maps = cond_maps_prev[j] if \
                        i < self.num_res_blocks // 2 else cond_maps_now[j]
                    x_img = getattr(self, 'res_' + str(i))(x_img, *cond_maps)

            # Optical flow warped image features.
            if warp_prev:
                # Estimate flow & mask.
                label_concat = torch.cat([label_prev.view(bs, -1, h, w),
                                          label], dim=1)
                img_prev_concat = img_prev.view(bs, -1, h, w)
                flow, mask = self.flow_network_temp(
                    label_concat, img_prev_concat)
                img_warp = resample(img_prev[:, -1], flow)
                if self.spade_combine:
                    # if using SPADE combine, integrate the warped image (and
                    # occlusion mask) into conditional inputs for SPADE.
                    img_embed = torch.cat([img_warp, mask], dim=1)
                    cond_maps_img = self.get_cond_maps(img_embed,
                                                       self.img_prev_embedding)
                    x_raw_img = None

            # Main image generation branch.
            for i in range(self.num_downsamples_img, -1, -1):
                # Get SPADE conditional inputs.
                j = min(i, self.num_downsamples_embed)
                cond_maps = cond_maps_now[j]

                # For raw output generation.
                if self.generate_raw_output:
                    if i >= self.num_multi_spade_layers - 1:
                        x_raw_img = x_img
                    if i < self.num_multi_spade_layers:
                        x_raw_img = self.one_up_conv_layer(
                            x_raw_img, cond_maps, i)

                # Add flow and guidance features.
                if warp_prev:
                    if i < self.num_multi_spade_layers:
                        # Add flow.
                        cond_maps += cond_maps_img[j]
                        # Add guidance.
                        if guidance_images_and_masks is not None:
                            cond_maps += [guidance_images_and_masks]
                    elif not self.guidance_only_with_flow:
                        # Add guidance if it is to be applied to every layer.
                        if guidance_images_and_masks is not None:
                            cond_maps += [guidance_images_and_masks]

                x_img = self.one_up_conv_layer(x_img, cond_maps, i)

            # Final conv layer.
            img_final = torch.tanh(self.conv_img(x_img))
            fake_images_source = 'in_training'

        # Update the point cloud color dict of renderer.
        self.renderer_update_point_cloud(img_final, point_info)

        output = dict()
        output['fake_images'] = img_final
        output['fake_flow_maps'] = flow
        output['fake_occlusion_masks'] = mask
        output['fake_raw_images'] = None
        output['warped_images'] = img_warp
        output['guidance_images_and_masks'] = guidance_images_and_masks
        output['fake_images_source'] = fake_images_source
        return output

    def get_cond_dims(self, num_downs=0):
        r"""Get the dimensions of conditional inputs.
        Args:
           num_downs (int) : How many downsamples at current layer.
        Returns:
           ch (list) : List of dimensions.
        """
        if not self.use_embed:
            ch = [self.num_input_channels]
        else:
            num_filters = getattr(self.emb_cfg, 'num_filters', 32)
            num_downs = min(num_downs, self.num_downsamples_embed)
            ch = [min(self.max_num_filters, num_filters * (2 ** num_downs))]
            if (num_downs < self.num_multi_spade_layers):
                ch = ch * 2
                # Also add guidance (RGB + mask = 4 channels, or 3 if partial).
                if self.guidance_partial_conv:
                    ch.append(3)
                else:
                    ch.append(4)
            elif not self.guidance_only_with_flow:
                if self.guidance_partial_conv:
                    ch.append(3)
                else:
                    ch.append(4)
        return ch

    def get_partial(self, num_downs=0):
        r"""Get if convs should be partial or not.
        Args:
           num_downs (int) : How many downsamples at current layer.
        Returns:
           partial (list) : List of boolean partial or not.
        """
        partial = [False]
        if (num_downs < self.num_multi_spade_layers):
            partial = partial * 2
            # Also add guidance (RGB + mask = 4 channels, or 3 if partial).
            if self.guidance_partial_conv:
                partial.append(True)
            else:
                partial.append(False)
        elif not self.guidance_only_with_flow:
            if self.guidance_partial_conv:
                partial.append(True)
            else:
                partial.append(False)
        return partial

    def get_cond_maps(self, label, embedder):
        r"""Get the conditional inputs.
        Args:
           label (4D tensor) : Input label tensor.
           embedder (obj) : Embedding network.
        Returns:
           cond_maps (list) : List of conditional inputs.
        """
        if not self.use_embed:
            return [label] * (self.num_layers + 1)
        embedded_label = embedder(label)
        cond_maps = [embedded_label]
        cond_maps = [[m[i] for m in cond_maps] for i in
                     range(len(cond_maps[0]))]
        return cond_maps
