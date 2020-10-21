# share: outside-ok
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import os
import time

import imageio
import numpy as np
import torch
from tqdm import tqdm

from imaginaire.losses import MaskedL1Loss
from imaginaire.model_utils.fs_vid2vid import concat_frames, resample
from imaginaire.trainers.vid2vid import Trainer as Vid2VidTrainer
from imaginaire.utils.distributed import is_master
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.misc import split_labels, to_cuda
from imaginaire.utils.visualization import tensor2flow, tensor2im


class Trainer(Vid2VidTrainer):
    r"""Initialize world consistent vid2vid trainer.

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
        self.guidance_start_after = getattr(cfg.gen.guidance, 'start_from', 0)
        self.train_data_loader = train_data_loader

    def _define_custom_losses(self):
        r"""All other custom losses are defined here."""
        # Setup the guidance loss.
        self.criteria['Guidance'] = MaskedL1Loss(normalize_over_valid=True)
        self.weights['Guidance'] = self.cfg.trainer.loss_weight.guidance

    def start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current iteration number.
        """
        self.net_G_module.reset_renderer(is_flipped_input=data['is_flipped'])
        # Keep unprojections on cpu to prevent unnecessary transfer.
        unprojections = data.pop('unprojections')
        data = to_cuda(data)
        data['unprojections'] = unprojections

        self.current_iteration = current_iteration
        if not self.is_inference:
            self.net_D.train()
        self.net_G.train()
        self.start_iteration_time = time.time()
        return data

    def reset(self):
        r"""Reset the trainer (for inference) at the beginning of a sequence."""
        # Inference time.
        self.net_G_module.reset_renderer(is_flipped_input=False)

        # print('Resetting trainer.')
        self.net_G_output = self.data_prev = None
        self.t = 0

        test_in_model_average_mode = getattr(
            self, 'test_in_model_average_mode', False)
        if test_in_model_average_mode:
            if hasattr(self.net_G.module.averaged_model, 'reset'):
                self.net_G.module.averaged_model.reset()
        else:
            if hasattr(self.net_G.module, 'reset'):
                self.net_G.module.reset()

    def create_sequence_output_dir(self, output_dir, key):
        r"""Create output subdir for this sequence.

        Args:
            output_dir (str): Root output dir.
            key (str): LMDB key which contains sequence name and file name.
            Returns:
            output_dir (str): Output subdir for this sequence.
            seq_name (str): Name of this sequence.
        """
        seq_dir = '/'.join(key.split('/')[:-1])
        output_dir = os.path.join(output_dir, seq_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + '/all', exist_ok=True)
        os.makedirs(output_dir + '/fake', exist_ok=True)
        seq_name = seq_dir.replace('/', '-')
        return output_dir, seq_name

    def test(self, test_data_loader, root_output_dir, inference_args):
        r"""Run inference on all sequences.

        Args:
            test_data_loader (object): Test data loader.
            root_output_dir (str): Location to dump outputs.
            inference_args (optional): Optional args.
        """

        # Go over all sequences.
        loader = test_data_loader
        num_inference_sequences = loader.dataset.num_inference_sequences()
        for sequence_idx in range(num_inference_sequences):
            loader.dataset.set_inference_sequence_idx(sequence_idx)
            print('Seq id: %d, Seq length: %d' %
                  (sequence_idx + 1, len(loader)))

            # Reset model at start of new inference sequence.
            self.reset()
            self.sequence_length = len(loader)

            # Go over all frames of this sequence.
            video = []
            for idx, data in enumerate(tqdm(loader)):
                key = data['key']['images'][0][0]
                filename = key.split('/')[-1]

                # Create output dir for this sequence.
                if idx == 0:
                    output_dir, seq_name = \
                        self.create_sequence_output_dir(root_output_dir, key)
                    video_path = os.path.join(output_dir, '..', seq_name)

                # Get output, and save all vis to all/.
                data['img_name'] = filename
                data = to_cuda(data)
                output = self.test_single(data, output_dir=output_dir + '/all')

                # Dump just the fake image here.
                fake = tensor2im(output['fake_images'])[0]
                video.append(fake)
                imageio.imsave(output_dir + '/fake/%s.jpg' % (filename), fake)

            # Save as mp4 and gif.
            imageio.mimsave(video_path + '.mp4', video, fps=15)

    def test_single(self, data, output_dir=None, save_fake_only=False):
        r"""The inference function. If output_dir exists, also save the
        output image.

        Args:
            data (dict): Training data at the current iteration.
            output_dir (str): Save image directory.
            save_fake_only (bool): Only save the fake output image.
        """
        if self.is_inference and self.cfg.trainer.model_average:
            test_in_model_average_mode = True
        else:
            test_in_model_average_mode = getattr(
                self, 'test_in_model_average_mode', False)
        data_t = self.get_data_t(data, self.net_G_output, self.data_prev, 0)
        if self.sequence_length > 1:
            self.data_prev = data_t

        # Generator forward.
        # Reset renderer if first time step.
        if self.t == 0:
            self.net_G_module.reset_renderer(
                is_flipped_input=data['is_flipped'])
        with torch.no_grad():
            if test_in_model_average_mode:
                net_G = self.net_G.module.averaged_model
            else:
                net_G = self.net_G
            self.net_G_output = net_G(data_t)

        if output_dir is not None:
            if save_fake_only:
                image_grid = tensor2im(self.net_G_output['fake_images'])[0]
            else:
                vis_images = self.get_test_output_images(data)
                image_grid = np.hstack([np.vstack(im) for im in
                                        vis_images if im is not None])
            if 'img_name' in data:
                save_name = data['img_name'].split('.')[0] + '.jpg'
            else:
                save_name = '%04d.jpg' % self.t
            output_filename = os.path.join(output_dir, save_name)
            os.makedirs(output_dir, exist_ok=True)
            imageio.imwrite(output_filename, image_grid)
            self.t += 1

        return self.net_G_output

    def get_test_output_images(self, data):
        r"""Get the visualization output of test function.

        Args:
            data (dict): Training data at the current iteration.
        """
        # Visualize labels.
        label_lengths = self.val_data_loader.dataset.get_label_lengths()
        labels = split_labels(data['label'], label_lengths)
        vis_labels = []
        for key, value in labels.items():
            if key == 'seg_maps':
                vis_labels.append(self.visualize_label(value[:, -1]))
            else:
                vis_labels.append(tensor2im(value[:, -1]))

        # Get gt image.
        im = tensor2im(data['images'][:, -1])

        # Get guidance image and masks.
        if self.net_G_output['guidance_images_and_masks'] is not None:
            guidance_image = tensor2im(
                self.net_G_output['guidance_images_and_masks'][:, :3])
            guidance_mask = tensor2im(
                self.net_G_output['guidance_images_and_masks'][:, 3:4],
                normalize=False)
        else:
            guidance_image = [np.zeros_like(item) for item in im]
            guidance_mask = [np.zeros_like(item) for item in im]

        # Create output.
        vis_images = [
            *vis_labels,
            im,
            guidance_image, guidance_mask,
            tensor2im(self.net_G_output['fake_images']),
        ]
        return vis_images

    def gen_frames(self, data, use_model_average=False):
        r"""Generate a sequence of frames given a sequence of data.

        Args:
            data (dict): Training data at the current iteration.
            use_model_average (bool): Whether to use model average
                for update or not.
        """
        net_G_output = None  # Previous generator output.
        data_prev = None  # Previous data.
        if use_model_average:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G

        # Iterate through the length of sequence.
        self.net_G_module.reset_renderer(is_flipped_input=data['is_flipped'])

        all_info = {'inputs': [], 'outputs': []}
        for t in range(self.sequence_length):
            # Get the data at the current time frame.
            data_t = self.get_data_t(data, net_G_output, data_prev, t)
            data_prev = data_t

            # Generator forward.
            with torch.no_grad():
                net_G_output = net_G(data_t)

            # Do any postprocessing if necessary.
            data_t, net_G_output = self.post_process(data_t, net_G_output)

            if t == 0:
                # Get the output at beginning of sequence for visualization.
                first_net_G_output = net_G_output

            all_info['inputs'].append(data_t)
            all_info['outputs'].append(net_G_output)

        return first_net_G_output, net_G_output, all_info

    def _get_custom_gen_losses(self, data_t, net_G_output, net_D_output):
        r"""All other custom generator losses go here.

        Args:
            data_t (dict): Training data at the current time t.
            net_G_output (dict): Output of the generator.
            net_D_output (dict): Output of the discriminator.
        """
        # Compute guidance loss.
        if net_G_output['guidance_images_and_masks'] is not None:
            guidance_image = net_G_output['guidance_images_and_masks'][:, :3]
            guidance_mask = net_G_output['guidance_images_and_masks'][:, 3:]
            self.gen_losses['Guidance'] = self.criteria['Guidance'](
                net_G_output['fake_images'], guidance_image, guidance_mask)
        else:
            self.gen_losses['Guidance'] = self.Tensor(1).fill_(0)

    def get_data_t(self, data, net_G_output, data_prev, t):
        r"""Get data at current time frame given the sequence of data.

        Args:
            data (dict): Training data for current iteration.
            net_G_output (dict): Output of the generator (for previous frame).
            data_prev (dict): Data for previous frame.
            t (int): Current time.
        """
        label = data['label'][:, t]
        image = data['images'][:, t]

        # Get keypoint mapping.
        unprojection = None
        if t >= self.guidance_start_after:
            if 'unprojections' in data:
                try:
                    # Remove unwanted padding.
                    unprojection = {}
                    for key, value in data['unprojections'].items():
                        value = value[0, t].cpu().numpy()
                        length = value[-1][0]
                        unprojection[key] = value[:length]
                except:  # noqa
                    pass

        if data_prev is not None:
            # Concat previous labels/fake images to the ones before.
            num_frames_G = self.cfg.data.num_frames_G
            prev_labels = concat_frames(data_prev['prev_labels'],
                                        data_prev['label'], num_frames_G - 1)
            prev_images = concat_frames(
                data_prev['prev_images'],
                net_G_output['fake_images'].detach(), num_frames_G - 1)
        else:
            prev_labels = prev_images = None

        data_t = dict()
        data_t['label'] = label
        data_t['image'] = image
        data_t['prev_labels'] = prev_labels
        data_t['prev_images'] = prev_images
        data_t['real_prev_image'] = data['images'][:, t - 1] if t > 0 else None
        data_t['unprojection'] = unprojection
        return data_t

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
            first_net_G_output, net_G_output, all_info = self.gen_frames(data)
            if self.cfg.trainer.model_average:
                first_net_G_output_avg, net_G_output_avg = self.gen_frames(
                    data, use_model_average=True)

        # Visualize labels.
        label_lengths = self.train_data_loader.dataset.get_label_lengths()
        labels = split_labels(data['label'], label_lengths)
        vis_labels_start, vis_labels_end = [], []
        for key, value in labels.items():
            if 'seg_maps' in key:
                vis_labels_start.append(self.visualize_label(value[:, -1]))
                vis_labels_end.append(self.visualize_label(value[:, 0]))
            else:
                normalize = self.train_data_loader.dataset.normalize[key]
                vis_labels_start.append(
                    tensor2im(value[:, -1], normalize=normalize))
                vis_labels_end.append(
                    tensor2im(value[:, 0], normalize=normalize))

        if is_master():
            vis_images = [
                *vis_labels_start,
                tensor2im(data['images'][:, -1]),
                tensor2im(net_G_output['fake_images']),
                tensor2im(net_G_output['fake_raw_images'])]
            if self.cfg.trainer.model_average:
                vis_images += [
                    tensor2im(net_G_output_avg['fake_images']),
                    tensor2im(net_G_output_avg['fake_raw_images'])]

            if self.sequence_length > 1:
                if net_G_output['guidance_images_and_masks'] is not None:
                    guidance_image = tensor2im(
                        net_G_output['guidance_images_and_masks'][:, :3])
                    guidance_mask = tensor2im(
                        net_G_output['guidance_images_and_masks'][:, 3:4],
                        normalize=False)
                else:
                    im = tensor2im(data['images'][:, -1])
                    guidance_image = [np.zeros_like(item) for item in im]
                    guidance_mask = [np.zeros_like(item) for item in im]
                vis_images += [guidance_image, guidance_mask]

                vis_images_first = [
                    *vis_labels_end,
                    tensor2im(data['images'][:, 0]),
                    tensor2im(first_net_G_output['fake_images']),
                    tensor2im(first_net_G_output['fake_raw_images']),
                    [np.zeros_like(item) for item in guidance_image],
                    [np.zeros_like(item) for item in guidance_mask]
                ]
                if self.cfg.trainer.model_average:
                    vis_images_first += [
                        tensor2im(first_net_G_output_avg['fake_images']),
                        tensor2im(first_net_G_output_avg['fake_raw_images'])]

                if self.use_flow:
                    flow_gt, conf_gt = self.criteria['Flow'].flowNet(
                        data['images'][:, -1], data['images'][:, -2])
                    warped_image_gt = resample(data['images'][:, -1], flow_gt)
                    vis_images_first += [
                        tensor2flow(flow_gt),
                        tensor2im(conf_gt, normalize=False),
                        tensor2im(warped_image_gt),
                    ]
                    vis_images += [
                        tensor2flow(net_G_output['fake_flow_maps']),
                        tensor2im(net_G_output['fake_occlusion_masks'],
                                  normalize=False),
                        tensor2im(net_G_output['warped_images']),
                    ]
                    if self.cfg.trainer.model_average:
                        vis_images_first += [
                            tensor2flow(flow_gt),
                            tensor2im(conf_gt, normalize=False),
                            tensor2im(warped_image_gt),
                        ]
                        vis_images += [
                            tensor2flow(net_G_output_avg['fake_flow_maps']),
                            tensor2im(net_G_output_avg['fake_occlusion_masks'],
                                      normalize=False),
                            tensor2im(net_G_output_avg['warped_images'])]

                vis_images = [[np.vstack((im_first, im))
                               for im_first, im in zip(imgs_first, imgs)]
                              for imgs_first, imgs in zip(vis_images_first,
                                                          vis_images)
                              if imgs is not None]

            image_grid = np.hstack([np.vstack(im) for im in
                                    vis_images if im is not None])

            print('Save output images to {}'.format(path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            imageio.imwrite(path, image_grid)

            # Gather all inputs and outputs for dumping into video.
            if self.sequence_length > 1:
                input_images, output_images, output_guidance = [], [], []
                for item in all_info['inputs']:
                    input_images.append(tensor2im(item['image'])[0])
                for item in all_info['outputs']:
                    output_images.append(tensor2im(item['fake_images'])[0])
                    if item['guidance_images_and_masks'] is not None:
                        output_guidance.append(tensor2im(
                            item['guidance_images_and_masks'][:, :3])[0])
                    else:
                        output_guidance.append(np.zeros_like(output_images[-1]))

                imageio.mimwrite(os.path.splitext(path)[0] + '.mp4',
                                 output_images, fps=2, macro_block_size=None)
                imageio.mimwrite(os.path.splitext(path)[0] + '_guidance.mp4',
                                 output_guidance, fps=2, macro_block_size=None)

            # for idx, item in enumerate(output_guidance):
            #     imageio.imwrite(os.path.splitext(
            #         path)[0] + '_guidance_%d.jpg' % (idx), item)
            # for idx, item in enumerate(input_images):
            #     imageio.imwrite(os.path.splitext(
            #         path)[0] + '_input_%d.jpg' % (idx), item)

        self.net_G.float()

    def _compute_fid(self):
        r"""Compute fid. Ignore for faster training."""
        return None

    def load_checkpoint(self, cfg, checkpoint_path):
        r"""Save network weights, optimizer parameters, scheduler parameters
        in the checkpoint.

        Args:
            cfg (obj): Global configuration.
            checkpoint_path (str): Path to the checkpoint.
        """
        # Create the single image model.
        if self.train_data_loader is None:
            load_single_image_model_weights = False
        else:
            load_single_image_model_weights = True
        self.net_G.module._init_single_image_model(
            load_weights=load_single_image_model_weights)

        # Call the original super function.
        return super().load_checkpoint(cfg, checkpoint_path)
