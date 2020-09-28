# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm


from imaginaire.model_utils.fs_vid2vid import (concat_frames, get_fg_mask,
                                               pre_process_densepose,
                                               random_roll)
from imaginaire.model_utils.pix2pixHD import get_optimizer_with_params
from imaginaire.trainers.vid2vid import Trainer as vid2vidTrainer
from imaginaire.utils.distributed import is_master
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.misc import to_cuda
from imaginaire.utils.visualization import tensor2flow, tensor2im


class Trainer(vid2vidTrainer):
    r"""Initialize vid2vid trainer.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self, cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                 train_data_loader, val_data_loader):
        super(Trainer, self).__init__(cfg, net_G, net_D, opt_G,
                                      opt_D, sch_G, sch_D,
                                      train_data_loader, val_data_loader)

    def _start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current number of iteration.
        """
        data = self.pre_process(data)
        return to_cuda(data)

    def pre_process(self, data):
        r"""Do any data pre-processing here.

        Args:
            data (dict): Data used for the current iteration.
        """
        data_cfg = self.cfg.data
        if hasattr(data_cfg, 'for_pose_dataset') and \
                ('pose_maps-densepose' in data_cfg.input_labels):
            pose_cfg = data_cfg.for_pose_dataset
            data['label'] = pre_process_densepose(pose_cfg, data['label'],
                                                  self.is_inference)
            data['few_shot_label'] = pre_process_densepose(
                pose_cfg, data['few_shot_label'], self.is_inference)
        return data

    def get_test_output_images(self, data):
        r"""Get the visualization output of test function.

        Args:
            data (dict): Training data at the current iteration.
        """
        vis_images = [
            tensor2im(data['few_shot_images'][:, 0]),
            self.visualize_label(data['label'][:, -1]),
            tensor2im(data['images'][:, -1]),
            tensor2im(self.net_G_output['fake_images']),
        ]
        return vis_images

    def get_data_t(self, data, net_G_output, data_prev, t):
        r"""Get data at current time frame given the sequence of data.

        Args:
            data (dict): Training data for current iteration.
            net_G_output (dict): Output of the generator (for previous frame).
            data_prev (dict): Data for previous frame.
            t (int): Current time.
        """
        label = data['label'][:, t] if 'label' in data else None
        image = data['images'][:, t]

        if data_prev is not None:
            nG = self.cfg.data.num_frames_G
            prev_labels = concat_frames(data_prev['prev_labels'],
                                        data_prev['label'], nG - 1)
            prev_images = concat_frames(
                data_prev['prev_images'],
                net_G_output['fake_images'].detach(), nG - 1)
        else:
            prev_labels = prev_images = None

        data_t = dict()
        data_t['label'] = label
        data_t['image'] = image
        data_t['ref_labels'] = data['few_shot_label'] if 'few_shot_label' \
                                                         in data else None
        data_t['ref_images'] = data['few_shot_images']
        data_t['prev_labels'] = prev_labels
        data_t['prev_images'] = prev_images
        data_t['real_prev_image'] = data['images'][:, t - 1] if t > 0 else None

        if 'landmarks_xy' in data:
            data_t['landmarks_xy'] = data['landmarks_xy'][:, t]
            data_t['ref_landmarks_xy'] = data['few_shot_landmarks_xy']
        return data_t

    def post_process(self, data, net_G_output):
        r"""Do any postprocessing of the data / output here.

        Args:
            data (dict): Training data at the current iteration.
            net_G_output (dict): Output of the generator.
        """
        if self.has_fg:
            fg_mask = get_fg_mask(data['label'], self.has_fg)
            if net_G_output['fake_raw_images'] is not None:
                net_G_output['fake_raw_images'] = \
                    net_G_output['fake_raw_images'] * fg_mask

        return data, net_G_output

    def test(self, test_data_loader, root_output_dir, inference_args):
        r"""Run inference on the specified sequence.

        Args:
            test_data_loader (object): Test data loader.
            root_output_dir (str): Location to dump outputs.
            inference_args (optional): Optional args.
        """
        self.reset()
        test_data_loader.dataset.set_sequence_length(0)
        # Set the inference sequences.
        test_data_loader.dataset.set_inference_sequence_idx(
            inference_args.driving_seq_index,
            inference_args.few_shot_seq_index,
            inference_args.few_shot_frame_index)

        video = []
        for idx, data in enumerate(tqdm(test_data_loader)):
            key = data['key']['images'][0][0]
            filename = key.split('/')[-1]

            # Create output dir for this sequence.
            if idx == 0:
                seq_name = '%03d' % inference_args.driving_seq_index
                output_dir = os.path.join(root_output_dir, seq_name)
                os.makedirs(output_dir, exist_ok=True)
                video_path = output_dir

            # Get output and save images.
            data['img_name'] = filename
            data = self.start_of_iteration(data, current_iteration=-1)
            output = self.test_single(data, output_dir, inference_args)
            video.append(output)

        # Save output as mp4.
        imageio.mimsave(video_path + '.mp4', video, fps=15)

    def save_image(self, path, data):
        r"""Save the output images to path.
        Note when the generate_raw_output is FALSE. Then,
        first_net_G_output['fake_raw_images'] is None and will not be displayed.
        In model average mode, we will plot the flow visualization twice.

        Args:
            path (str): Save path.
            data (dict): Training data for current iteration.
        """
        self.net_G.eval()
        if self.cfg.trainer.model_average:
            self.net_G.module.averaged_model.eval()

        self.net_G_output = None
        with torch.no_grad():
            first_net_G_output, last_net_G_output, _ = self.gen_frames(data)
            if self.cfg.trainer.model_average:
                first_net_G_output_avg, last_net_G_output_avg, _ = \
                    self.gen_frames(data, use_model_average=True)

        def get_images(data, net_G_output, return_first_frame=True,
                       for_model_average=False):
            r"""Get the ourput images to save.

            Args:
                data (dict): Training data for current iteration.
                net_G_output (dict): Generator output.
                return_first_frame (bool): Return output for first frame in the
                sequence.
                for_model_average (bool): For model average output.
            Return:
                vis_images (list of numpy arrays): Visualization images.
            """
            frame_idx = 0 if return_first_frame else -1
            warped_idx = 0 if return_first_frame else 1
            vis_images = []
            if not for_model_average:
                vis_images += [
                    tensor2im(data['few_shot_images'][:, frame_idx]),
                    self.visualize_label(data['label'][:, frame_idx]),
                    tensor2im(data['images'][:, frame_idx])
                ]
            vis_images += [
                tensor2im(net_G_output['fake_images']),
                tensor2im(net_G_output['fake_raw_images'])]
            if not for_model_average:
                vis_images += [
                    tensor2im(net_G_output['warped_images'][warped_idx]),
                    tensor2flow(net_G_output['fake_flow_maps'][warped_idx]),
                    tensor2im(net_G_output['fake_occlusion_masks'][warped_idx],
                              normalize=False)
                ]
            return vis_images

        if is_master():
            vis_images_first = get_images(data, first_net_G_output)
            if self.cfg.trainer.model_average:
                vis_images_first += get_images(data, first_net_G_output_avg,
                                               for_model_average=True)
            if self.sequence_length > 1:
                vis_images_last = get_images(data, last_net_G_output,
                                             return_first_frame=False)
                if self.cfg.trainer.model_average:
                    vis_images_last += get_images(data, last_net_G_output_avg,
                                                  return_first_frame=False,
                                                  for_model_average=True)

                # If generating a video, the first row of each batch will be
                # the first generated frame and the flow/mask for warping the
                # reference image, and the second row will be the last
                # generated frame and the flow/mask for warping the previous
                # frame. If using model average, the frames generated by model
                # average will be at the rightmost columns.
                vis_images = [[np.vstack((im_first, im_last))
                               for im_first, im_last in
                               zip(imgs_first, imgs_last)]
                              for imgs_first, imgs_last in zip(vis_images_first,
                                                               vis_images_last)
                              if imgs_first is not None]
            else:
                vis_images = vis_images_first

            image_grid = np.hstack([np.vstack(im) for im in vis_images
                                    if im is not None])

            print('Save output images to {}'.format(path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            imageio.imwrite(path, image_grid)

    def finetune(self, data, inference_args):
        r"""Finetune the model for a few iterations on the inference data."""
        # Get the list of params to finetune.
        self.net_G, self.net_D, self.opt_G, self.opt_D = \
            get_optimizer_with_params(self.cfg, self.net_G, self.net_D,
                                      param_names_start_with=[
                                          'weight_generator.fc', 'conv_img',
                                          'up'])
        data_finetune = {k: v for k, v in data.items()}
        ref_labels = data_finetune['few_shot_label']
        ref_images = data_finetune['few_shot_images']

        # Number of iterations to finetune.
        iterations = getattr(inference_args, 'finetune_iter', 100)
        for it in range(1, iterations + 1):
            # Randomly set one of the reference images as target.
            idx = np.random.randint(ref_labels.size(1))
            tgt_label, tgt_image = ref_labels[:, idx], ref_images[:, idx]
            # Randomly shift and flip the target image.
            tgt_label, tgt_image = random_roll([tgt_label, tgt_image])
            data_finetune['label'] = tgt_label.unsqueeze(1)
            data_finetune['images'] = tgt_image.unsqueeze(1)

            self.gen_update(data_finetune)
            self.dis_update(data_finetune)
            if (it % (iterations // 10)) == 0:
                print(it)

        self.has_finetuned = True
