# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os
import time

import torch
import torchvision
from torch import nn
from tqdm import tqdm

from apex import amp
from imaginaire.utils.distributed import is_master, master_only
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.io import save_pilimage_in_jpeg
from imaginaire.utils.meters import Meter, add_hparams
from imaginaire.utils.misc import to_cuda, to_device, requires_grad
from imaginaire.utils.model_average import (calibrate_batch_norm_momentum,
                                            reset_batch_norm)
from imaginaire.utils.visualization import tensor2pilimage


class BaseTrainer(object):
    r"""Base trainer. We expect that all trainers inherit this class.

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

    def __init__(self,
                 cfg,
                 net_G,
                 net_D,
                 opt_G,
                 opt_D,
                 sch_G,
                 sch_D,
                 train_data_loader,
                 val_data_loader):
        super(BaseTrainer, self).__init__()
        print('Setup trainer.')

        # Initialize models and data loaders.
        self.cfg = cfg
        self.net_G = net_G
        if cfg.trainer.model_average:
            # Two wrappers (DDP + model average).
            self.net_G_module = self.net_G.module.module
        else:
            # One wrapper (DDP)
            self.net_G_module = self.net_G.module
        self.val_data_loader = val_data_loader
        self.is_inference = train_data_loader is None
        self.net_D = net_D
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.sch_G = sch_G
        self.sch_D = sch_D
        self.train_data_loader = train_data_loader

        # Initialize loss functions.
        # All loss names have weights. Some have criterion modules.
        # Mapping from loss names to criterion modules.
        self.criteria = nn.ModuleDict()
        # Mapping from loss names to loss weights.
        self.weights = dict()
        self.losses = dict(gen_update=dict(), dis_update=dict())
        self.gen_losses = self.losses['gen_update']
        self.dis_losses = self.losses['dis_update']
        self._init_loss(cfg)
        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))
            if loss_name in self.criteria.keys() and \
                    self.criteria[loss_name] is not None:
                self.criteria[loss_name].to('cuda')

        if self.is_inference:
            # The initialization steps below can be skipped during inference.
            return

        # Initialize logging attributes.
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_iteration_time = None
        self.start_epoch_time = None
        self.elapsed_iteration_time = 0
        self.time_iteration = -1
        self.time_epoch = -1
        self.best_fid = None
        if getattr(self.cfg, 'speed_benchmark', False):
            self.accu_gen_forw_iter_time = 0
            self.accu_gen_loss_iter_time = 0
            self.accu_gen_back_iter_time = 0
            self.accu_gen_step_iter_time = 0
            self.accu_gen_avg_iter_time = 0
            self.accu_dis_forw_iter_time = 0
            self.accu_dis_loss_iter_time = 0
            self.accu_dis_back_iter_time = 0
            self.accu_dis_step_iter_time = 0

        # Initialize tensorboard and hparams.
        self._init_tensorboard()
        self._init_hparams()

    def _init_tensorboard(self):
        r"""Initialize the tensorboard. Different algorithms might require
        different performance metrics. Hence, custom tensorboard
        initialization might be necessary.
        """
        # Logging frequency: self.cfg.logging_iter
        self.meters = {}
        names = ['optim/gen_lr', 'optim/dis_lr', 'time/iteration', 'time/epoch']
        for name in names:
            self.meters[name] = Meter(name)

        # Logging frequency: self.cfg.snapshot_save_iter
        names = ['FID', 'best_FID']
        self.metric_meters = {}
        for name in names:
            self.metric_meters[name] = Meter(name)

        # Logging frequency: self.cfg.image_display_iter
        self.image_meter = Meter('images')

    def _init_hparams(self):
        r"""Initialize a dictionary of hyperparameters that we want to monitor
        in the HParams dashboard in tensorBoard.
        """
        self.hparam_dict = {}

    def _write_tensorboard(self):
        r"""Write values to tensorboard. By default, we will log the time used
        per iteration, time used per epoch, generator learning rate, and
        discriminator learning rate. We will log all the losses as well as
        custom meters.
        """
        # Logs that are shared by all models.
        self._write_to_meters({'time/iteration': self.time_iteration,
                               'time/epoch': self.time_epoch,
                               'optim/gen_lr': self.sch_G.get_last_lr()[0],
                               'optim/dis_lr': self.sch_D.get_last_lr()[0]},
                              self.meters)
        # Logs for loss values. Different models have different losses.
        self._write_loss_meters()
        # Other custom logs.
        self._write_custom_meters()

        # Write all logs to tensorboard.
        self._flush_meters(self.meters)

    def _write_loss_meters(self):
        r"""Write all loss values to tensorboard."""
        for update, losses in self.losses.items():
            # update is 'gen_update' or 'dis_update'.
            assert update == 'gen_update' or update == 'dis_update'
            for loss_name, loss in losses.items():
                full_loss_name = update + '/' + loss_name
                if full_loss_name not in self.meters.keys():
                    # Create a new meter if it doesn't exist.
                    self.meters[full_loss_name] = Meter(full_loss_name)
                self.meters[full_loss_name].write(loss.item())

    def _write_custom_meters(self):
        r"""Dummy member function to be overloaded by the child class.
        In the child class, you can write down whatever you want to track.
        """
        pass

    @staticmethod
    def _write_to_meters(data, meters):
        r"""Write values to meters."""
        for key, value in data.items():
            meters[key].write(value)

    def _flush_meters(self, meters):
        r"""Flush all meters using the current iteration."""
        for meter in meters.values():
            meter.flush(self.current_iteration)

    def _pre_save_checkpoint(self):
        r"""Implement the things you want to do before saving a checkpoint.
        For example, you can compute the K-mean features (pix2pixHD) before
        saving the model weights to a checkpoint.
        """
        pass

    def save_checkpoint(self, current_epoch, current_iteration):
        r"""Save network weights, optimizer parameters, scheduler parameters
        to a checkpoint.
        """
        self._pre_save_checkpoint()
        _save_checkpoint(self.cfg,
                         self.net_G, self.net_D,
                         self.opt_G, self.opt_D,
                         self.sch_G, self.sch_D,
                         current_epoch, current_iteration)

    def load_checkpoint(self, cfg, checkpoint_path, resume=None):
        r"""Load network weights, optimizer parameters, scheduler parameters
        from a checkpoint.

        Args:
            cfg (obj): Global configuration.
            checkpoint_path (str): Path to the checkpoint.
            resume (bool or None): If not ``None``, will determine whether or
                not to load optimizers in addition to network weights.
        """
        if os.path.exists(checkpoint_path):
            # If checkpoint_path exists, we will load its weights to
            # initialize our network.
            if resume is None:
                resume = False
        elif os.path.exists(os.path.join(cfg.logdir, 'latest_checkpoint.txt')):
            # This is for resuming the training from the previously saved
            # checkpoint.
            fn = os.path.join(cfg.logdir, 'latest_checkpoint.txt')
            with open(fn, 'r') as f:
                line = f.read().splitlines()
            checkpoint_path = os.path.join(cfg.logdir, line[0].split(' ')[-1])
            if resume is None:
                resume = True
        else:
            # checkpoint not found and not specified. We will train
            # everything from scratch.
            current_epoch = 0
            current_iteration = 0
            print('No checkpoint found.')
            return current_epoch, current_iteration
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
        current_epoch = 0
        current_iteration = 0
        if resume:
            self.net_G.load_state_dict(checkpoint['net_G'])
            if not self.is_inference:
                self.net_D.load_state_dict(checkpoint['net_D'])
                if 'opt_G' in checkpoint:
                    self.opt_G.load_state_dict(checkpoint['opt_G'])
                    self.opt_D.load_state_dict(checkpoint['opt_D'])
                    self.sch_G.load_state_dict(checkpoint['sch_G'])
                    self.sch_D.load_state_dict(checkpoint['sch_D'])
                    current_epoch = checkpoint['current_epoch']
                    current_iteration = checkpoint['current_iteration']
                    print('Load from: {}'.format(checkpoint_path))
                else:
                    print('Load network weights only.')
        else:
            self.net_G.load_state_dict(checkpoint['net_G'])
            print('Load generator weights only.')

        print('Done with loading the checkpoint.')
        return current_epoch, current_iteration

    def start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        self._start_of_epoch(current_epoch)
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()

    def start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current number of iteration.
        """
        data = self._start_of_iteration(data, current_iteration)
        data = to_cuda(data)
        self.current_iteration = current_iteration
        if not self.is_inference:
            self.net_D.train()
        self.net_G.train()
        # torch.cuda.synchronize()
        self.start_iteration_time = time.time()
        return data

    def end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Things to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        # Update the learning rate policy for the generator if operating in the
        # iteration mode.
        if self.cfg.gen_opt.lr_policy.iteration_mode:
            self.sch_G.step()
        # Update the learning rate policy for the discriminator if operating
        # in the iteration mode.
        if self.cfg.dis_opt.lr_policy.iteration_mode:
            self.sch_D.step()

        # Accumulate time
        # torch.cuda.synchronize()
        self.elapsed_iteration_time += time.time() - self.start_iteration_time
        # Logging.
        if current_iteration % self.cfg.logging_iter == 0:
            ave_t = self.elapsed_iteration_time / self.cfg.logging_iter
            self.time_iteration = ave_t
            print('Iteration: {}, average iter time: '
                  '{:6f}.'.format(current_iteration, ave_t))
            self.elapsed_iteration_time = 0

            if getattr(self.cfg, 'speed_benchmark', False):
                # Below code block only needed when analyzing computation
                # bottleneck.
                print('\tGenerator FWD time {:6f}'.format(
                    self.accu_gen_forw_iter_time / self.cfg.logging_iter))
                print('\tGenerator LOS time {:6f}'.format(
                    self.accu_gen_loss_iter_time / self.cfg.logging_iter))
                print('\tGenerator BCK time {:6f}'.format(
                    self.accu_gen_back_iter_time / self.cfg.logging_iter))
                print('\tGenerator STP time {:6f}'.format(
                    self.accu_gen_step_iter_time / self.cfg.logging_iter))
                print('\tGenerator AVG time {:6f}'.format(
                    self.accu_gen_avg_iter_time / self.cfg.logging_iter))

                print('\tDiscriminator FWD time {:6f}'.format(
                    self.accu_dis_forw_iter_time / self.cfg.logging_iter))
                print('\tDiscriminator LOS time {:6f}'.format(
                    self.accu_dis_loss_iter_time / self.cfg.logging_iter))
                print('\tDiscriminator BCK time {:6f}'.format(
                    self.accu_dis_back_iter_time / self.cfg.logging_iter))
                print('\tDiscriminator STP time {:6f}'.format(
                    self.accu_dis_step_iter_time / self.cfg.logging_iter))

                print('{:6f}'.format(ave_t))

                self.accu_gen_forw_iter_time = 0
                self.accu_gen_loss_iter_time = 0
                self.accu_gen_back_iter_time = 0
                self.accu_gen_step_iter_time = 0
                self.accu_gen_avg_iter_time = 0
                self.accu_dis_forw_iter_time = 0
                self.accu_dis_loss_iter_time = 0
                self.accu_dis_back_iter_time = 0
                self.accu_dis_step_iter_time = 0

        self._end_of_iteration(data, current_epoch, current_iteration)
        # Save everything to the checkpoint.
        if current_iteration >= self.cfg.snapshot_save_start_iter and \
                current_iteration % self.cfg.snapshot_save_iter == 0:
            self.save_image(self._get_save_path('images', 'jpg'), data)
            self.save_checkpoint(current_epoch, current_iteration)
            self.write_metrics()
        # Compute image to be saved.
        elif current_iteration % self.cfg.image_save_iter == 0:
            self.save_image(self._get_save_path('images', 'jpg'), data)
        elif current_iteration % self.cfg.image_display_iter == 0:
            image_path = os.path.join(self.cfg.logdir, 'images', 'current.jpg')
            self.save_image(image_path, data)
        if current_iteration % self.cfg.logging_iter == 0:
            self._write_tensorboard()

    def end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Things to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.

            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        # Update the learning rate policy for the generator if operating in the
        # epoch mode.
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        if not self.cfg.gen_opt.lr_policy.iteration_mode:
            self.sch_G.step()
        # Update the learning rate policy for the discriminator if operating
        # in the epoch mode.
        if not self.cfg.dis_opt.lr_policy.iteration_mode:
            self.sch_D.step()
        elapsed_epoch_time = time.time() - self.start_epoch_time
        # Logging.
        print('Epoch: {}, total time: {:6f}.'.format(current_epoch,
                                                     elapsed_epoch_time))
        self.time_epoch = elapsed_epoch_time
        self._end_of_epoch(data, current_epoch, current_iteration)
        # Save everything to the checkpoint.
        if current_epoch >= self.cfg.snapshot_save_start_epoch and \
                current_epoch % self.cfg.snapshot_save_epoch == 0:
            self.save_image(self._get_save_path('images', 'jpg'), data)
            self.save_checkpoint(current_epoch, current_iteration)
            self.write_metrics()

    def pre_process(self, data):
        r"""Custom data pre-processing function. Utilize this function if you
        need to preprocess your data before sending it to the generator and
        discriminator.

        Args:
            data (dict): Data used for the current iteration.
        """

    def recalculate_model_average_batch_norm_statistics(self, data_loader):
        r"""Update the statistics in the moving average model.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for
                estimating the statistics.
        """
        if not self.cfg.trainer.model_average:
            return
        model_average_iteration = \
            self.cfg.trainer.model_average_batch_norm_estimation_iteration
        if model_average_iteration == 0:
            return
        with torch.no_grad():
            # Accumulate bn stats..
            self.net_G.module.averaged_model.train()
            # Reset running stats.
            self.net_G.module.averaged_model.apply(reset_batch_norm)
            for cal_it, cal_data in enumerate(data_loader):
                if cal_it >= model_average_iteration:
                    print('Done with {} iterations of updating batch norm '
                          'statistics'.format(model_average_iteration))
                    break
                cal_data = to_device(cal_data, 'cuda')
                # Averaging over all batches
                self.net_G.module.averaged_model.apply(
                    calibrate_batch_norm_momentum)
                self.net_G.module.averaged_model(cal_data)

    def save_image(self, path, data):
        r"""Compute visualization images and save them to the disk.

        Args:
            path (str): Location of the file.
            data (dict): Data used for the current iteration.
        """
        self.net_G.eval()
        vis_images = self._get_visualizations(data)
        if is_master() and vis_images is not None:
            vis_images = torch.cat(vis_images, dim=3).float()
            vis_images = (vis_images + 1) / 2
            print('Save output images to {}'.format(path))
            vis_images.clamp_(0, 1)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image_grid = torchvision.utils.make_grid(
                vis_images, nrow=1, padding=0, normalize=False)
            if self.cfg.trainer.image_to_tensorboard:
                self.image_meter.write_image(image_grid, self.current_iteration)
            torchvision.utils.save_image(image_grid, path, nrow=1)

    def write_metrics(self):
        r"""Write metrics to the tensorboard."""
        cur_fid = self._compute_fid()
        if cur_fid is not None:
            if self.best_fid is not None:
                self.best_fid = min(self.best_fid, cur_fid)
            else:
                self.best_fid = cur_fid
            metric_dict = {'FID': cur_fid, 'best_FID': self.best_fid}
            self._write_to_meters(metric_dict, self.metric_meters)
            self._flush_meters(self.metric_meters)
            if self.cfg.trainer.hparam_to_tensorboard:
                add_hparams(self.hparam_dict, metric_dict)

    def _get_save_path(self, subdir, ext):
        r"""Get the image save path.

        Args:
            subdir (str): Sub-directory under the main directory for saving
                the outputs.
            ext (str): Filename extension for the image (e.g., jpg, png, ...).
        Return:
            (str): image filename to be used to save the visualization results.
        """
        subdir_path = os.path.join(self.cfg.logdir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(
            subdir_path, 'epoch_{:05}_iteration_{:09}.{}'.format(
                self.current_epoch, self.current_iteration, ext))

    def _get_outputs(self, net_D_output, real=True):
        r"""Return output values. Note that when the gan mode is relativistic.
        It will do the difference before returning.

        Args:
           net_D_output (dict):
               real_outputs (tensor): Real output values.
               fake_outputs (tensor): Fake output values.
           real (bool): Return real or fake.
        """

        def _get_difference(a, b):
            r"""Get difference between two lists of tensors or two tensors.

            Args:
                a: list of tensors or tensor
                b: list of tensors or tensor
            """
            out = list()
            for x, y in zip(a, b):
                if isinstance(x, list):
                    res = _get_difference(x, y)
                else:
                    res = x - y
                out.append(res)
            return out

        if real:
            if self.cfg.trainer.gan_relativistic:
                return _get_difference(net_D_output['real_outputs'],
                                       net_D_output['fake_outputs'])
            else:
                return net_D_output['real_outputs']
        else:
            if self.cfg.trainer.gan_relativistic:
                return _get_difference(net_D_output['fake_outputs'],
                                       net_D_output['real_outputs'])
            else:
                return net_D_output['fake_outputs']

    def _start_of_epoch(self, current_epoch):
        r"""Operations to do before starting an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        pass

    def _start_of_iteration(self, data, current_iteration):
        r"""Operations to do before starting an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current epoch number.
        Returns:
            (dict): Data used for the current iteration. They might be
                processed by the custom _start_of_iteration function.
        """
        return data

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Operations to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Operations to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _get_visualizations(self, data):
        r"""Compute visualization outputs.

        Args:
            data (dict): Data used for the current iteration.
        """
        return None

    def _compute_fid(self):
        r"""FID computation function to be overloaded."""
        return None

    def _init_loss(self, cfg):
        r"""Every trainer should implement its own init loss function."""
        raise NotImplementedError

    def gen_update(self, data):
        r"""Update the generator.

        Args:
            data (dict): Data used for the current iteration.
        """
        self.opt_G.zero_grad()

        # Set requires_grad flags.
        requires_grad(self.net_G_module, True)
        requires_grad(self.net_D, False)

        # Compute the loss.
        self._time_before_forward()
        total_loss = self.gen_forward(data)
        if total_loss is None:
            return

        # Backpropagate the loss.
        self._time_before_backward()
        with amp.scale_loss(total_loss, self.opt_G, loss_id=0) as scaled_loss:
            scaled_loss.backward()

        # Optionally clip gradient norm.
        if hasattr(self.cfg.gen_opt, 'clip_grad_norm'):
            nn.utils.clip_grad_norm_(
                amp.master_params(self.opt_G), self.cfg.gen_opt.clip_grad_norm)

        # Perform an optimizer step.
        self._time_before_step()
        self.opt_G.step()

        # Update model average.
        self._time_before_model_avg()
        if self.cfg.trainer.model_average:
            self.net_G.module.update_average()

        self._detach_losses()
        self._time_before_leave_gen()

    def gen_forward(self, data):
        r"""Every trainer should implement its own generator forward."""
        raise NotImplementedError

    def dis_update(self, data):
        r"""Update the discriminator.

        Args:
            data (dict): Data used for the current iteration.
        """
        self.opt_D.zero_grad()

        # Set requires_grad flags.
        requires_grad(self.net_G_module, False)
        requires_grad(self.net_D, True)

        # Compute the loss.
        self._time_before_forward()
        total_loss = self.dis_forward(data)
        if total_loss is None:
            return

        # Backpropagate the loss.
        self._time_before_backward()
        with amp.scale_loss(total_loss, self.opt_D, loss_id=1) as scaled_loss:
            scaled_loss.backward()

        # Perform an optimizer step.
        self._time_before_step()
        self.opt_D.step()

        self._detach_losses()
        self._time_before_leave_dis()

    def dis_forward(self, data):
        r"""Every trainer should implement its own discriminator forward."""
        raise NotImplementedError

    def test(self, data_loader, output_dir, inference_args):
        r"""Compute results images for a batch of input data and save the
        results in the specified folder.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch dataloader.
            output_dir (str): Target location for saving the output image.
        """
        if self.cfg.trainer.model_average:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        print('# of samples %d' % len(data_loader))
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                output_images, file_names = \
                    net_G.inference(data, **vars(inference_args))
            for output_image, file_name in zip(output_images, file_names):
                fullname = os.path.join(output_dir, file_name + '.jpg')
                output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                               minus1to1_normalized=True)
                save_pilimage_in_jpeg(fullname, output_image)

    def _get_total_loss(self, gen_forward):
        r"""Return the total loss to be backpropagated.
        Args:
            gen_forward (bool): If ``True``, backpropagates the generator loss,
                otherwise the discriminator loss.
        """
        losses = self.gen_losses if gen_forward else self.dis_losses
        total_loss = torch.tensor(0., device=torch.device('cuda'))
        # Iterates over all possible losses.
        for loss_name in self.weights:
            # If it is for the current model (gen/dis).
            if loss_name in losses:
                # Multiply it with the corresponding weight
                # and add it to the total loss.
                total_loss += losses[loss_name] * self.weights[loss_name]
        losses['total'] = total_loss  # logging purpose
        return total_loss

    def _detach_losses(self):
        r"""Detach all logging variables to prevent potential memory leak."""
        for loss_name in self.gen_losses:
            self.gen_losses[loss_name] = self.gen_losses[loss_name].detach()
        for loss_name in self.dis_losses:
            self.dis_losses[loss_name] = self.dis_losses[loss_name].detach()

    def _time_before_forward(self):
        r"""
        Record time before applying forward.
        """
        if getattr(self.cfg, 'speed_benchmark', False):
            torch.cuda.synchronize()
            self.forw_time = time.time()

    def _time_before_loss(self):
        r"""
        Record time before computing loss.
        """
        if getattr(self.cfg, 'speed_benchmark', False):
            torch.cuda.synchronize()
            self.loss_time = time.time()

    def _time_before_backward(self):
        r"""
        Record time before applying backward.
        """
        if getattr(self.cfg, 'speed_benchmark', False):
            torch.cuda.synchronize()
            self.back_time = time.time()

    def _time_before_step(self):
        r"""
        Record time before updating the weights
        """
        if getattr(self.cfg, 'speed_benchmark', False):
            torch.cuda.synchronize()
            self.step_time = time.time()

    def _time_before_model_avg(self):
        r"""
        Record time before applying model average.
        """
        if getattr(self.cfg, 'speed_benchmark', False):
            torch.cuda.synchronize()
            self.avg_time = time.time()

    def _time_before_leave_gen(self):
        r"""
        Record forward, backward, loss, and model average time for the
        generator update.
        """
        if getattr(self.cfg, 'speed_benchmark', False):
            torch.cuda.synchronize()
            end_time = time.time()
            self.accu_gen_forw_iter_time += self.loss_time - self.forw_time
            self.accu_gen_loss_iter_time += self.back_time - self.loss_time
            self.accu_gen_back_iter_time += self.step_time - self.back_time
            self.accu_gen_step_iter_time += self.avg_time - self.step_time
            self.accu_gen_avg_iter_time += end_time - self.avg_time

    def _time_before_leave_dis(self):
        r"""
        Record forward, backward, loss time for the discriminator update.
        """
        if getattr(self.cfg, 'speed_benchmark', False):
            torch.cuda.synchronize()
            end_time = time.time()
            self.accu_dis_forw_iter_time += self.loss_time - self.forw_time
            self.accu_dis_loss_iter_time += self.back_time - self.loss_time
            self.accu_dis_back_iter_time += self.step_time - self.back_time
            self.accu_dis_step_iter_time += end_time - self.step_time


@master_only
def _save_checkpoint(cfg,
                     net_G, net_D,
                     opt_G, opt_D,
                     sch_G, sch_D,
                     current_epoch, current_iteration):
    r"""Save network weights, optimizer parameters, scheduler parameters
    in the checkpoint.

    Args:
        cfg (obj): Global configuration.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        current_epoch (int): Current epoch.
        current_iteration (int): Current iteration.
    """
    latest_checkpoint_path = 'epoch_{:05}_iteration_{:09}_checkpoint.pt'.format(
        current_epoch, current_iteration)
    save_path = os.path.join(cfg.logdir, latest_checkpoint_path)
    torch.save(
        {
            'net_G': net_G.state_dict(),
            'net_D': net_D.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'sch_G': sch_G.state_dict(),
            'sch_D': sch_D.state_dict(),
            'current_epoch': current_epoch,
            'current_iteration': current_iteration,
        },
        save_path,
    )
    fn = os.path.join(cfg.logdir, 'latest_checkpoint.txt')
    with open(fn, 'wt') as f:
        f.write('latest_checkpoint: %s' % latest_checkpoint_path)
    print('Save checkpoint to {}'.format(save_path))
    return save_path
