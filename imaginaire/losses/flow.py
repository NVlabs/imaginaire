# share: outside-ok
# flake8: noqa
import importlib
import warnings

import torch
import torch.nn as nn

from imaginaire.model_utils.fs_vid2vid import (get_face_mask, get_fg_mask,
                                               get_part_mask, pick_image,
                                               resample)


class MaskedL1Loss(nn.Module):
    r"""Masked L1 loss constructor."""

    def __init__(self, normalize_over_valid=False):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.normalize_over_valid = normalize_over_valid

    def forward(self, input, target, mask):
        r"""Masked L1 loss computation.

        Args:
            input (tensor): Input tensor.
            target (tensor): Target tensor.
            mask (tensor): Mask to be applied to the output loss.

        Returns:
            (tensor): Loss value.
        """
        mask = mask.expand_as(input)
        loss = self.criterion(input * mask, target * mask)
        if self.normalize_over_valid:
            # The loss has been averaged over all pixels.
            # Only average over regions which are valid.
            loss = loss * torch.numel(mask) / (torch.sum(mask) + 1e-6)
        return loss


class FlowLoss(nn.Module):
    r"""Flow loss constructor.

    Args:
        cfg (obj): Configuration.
    """

    def __init__(self, cfg):
        super(FlowLoss, self).__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.criterion = nn.L1Loss()
        self.criterionMasked = MaskedL1Loss()
        flow_module = importlib.import_module(cfg.flow_network.type)
        # Automatically casting the flow network to half precision when using
        # ampO1.
        fp16 = cfg.trainer.amp > 'O0'
        self.flowNet = flow_module.FlowNet(pretrained=True, fp16=fp16)
        self.warp_ref = getattr(cfg.gen.flow, 'warp_ref', False)
        self.pose_cfg = pose_cfg = getattr(cfg.data, 'for_pose_dataset', None)
        self.for_pose_dataset = pose_cfg is not None
        self.has_fg = getattr(cfg.data, 'has_foreground', False)

    def forward(self, data, net_G_output, current_epoch):
        r"""Compute losses on the output flow and occlusion mask.

        Args:
            data (dict): Input data.
            net_G_output (dict): Generator output.
            current_epoch (int): Current training epoch number.
        Returns:
            (dict):
              - loss_flow_L1 (tensor): L1 loss compared to ground truth flow.
              - loss_flow_warp (tensor): L1 loss between the warped image and the
                target image when using the flow to warp.
              - loss_mask (tensor): Loss for the occlusion mask.
        """
        tgt_label, tgt_image = data['label'], data['image']

        fake_image = net_G_output['fake_images']
        warped_images = net_G_output['warped_images']
        flow = net_G_output['fake_flow_maps']
        occ_mask = net_G_output['fake_occlusion_masks']

        if self.warp_ref:
            # Pick the most similar reference image to warp.
            ref_labels, ref_images = data['ref_labels'], data['ref_images']
            ref_idx = net_G_output['ref_idx']
            ref_label, ref_image = pick_image([ref_labels, ref_images], ref_idx)
        else:
            ref_label = ref_image = None

        # Compute the ground truth flows and confidence maps.
        flow_gt_prev = flow_gt_ref = conf_gt_prev = conf_gt_ref = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.warp_ref:
                # Compute GT for warping reference -> target.
                if self.for_pose_dataset:
                    # Use DensePose maps to compute flows for pose dataset.
                    flow_gt_ref, conf_gt_ref = self.flowNet(tgt_label[:, :3],
                                                            ref_label[:, :3])
                else:
                    # Use RGB images for other datasets.
                    flow_gt_ref, conf_gt_ref = self.flowNet(tgt_image,
                                                            ref_image)

            if current_epoch >= self.cfg.single_frame_epoch and \
                    data['real_prev_image'] is not None:
                # Compute GT for warping previous -> target.
                tgt_image_prev = data['real_prev_image']
                flow_gt_prev, conf_gt_prev = self.flowNet(tgt_image,
                                                          tgt_image_prev)

        flow_gt = [flow_gt_ref, flow_gt_prev]
        flow_conf_gt = [conf_gt_ref, conf_gt_prev]
        # Get the foreground masks.
        fg_mask, ref_fg_mask = get_fg_mask([tgt_label, ref_label], self.has_fg)

        # Compute losses for flow maps and masks.
        loss_flow_L1, loss_flow_warp, body_mask_diff = \
            self.compute_flow_losses(flow, warped_images, tgt_image, flow_gt,
                                     flow_conf_gt, fg_mask, tgt_label,
                                     ref_label)

        loss_mask = self.compute_mask_losses(
            occ_mask, fake_image, warped_images, tgt_label, tgt_image,
            fg_mask, ref_fg_mask, body_mask_diff)

        return loss_flow_L1, loss_flow_warp, loss_mask

    def compute_flow_losses(self, flow, warped_images, tgt_image, flow_gt,
                            flow_conf_gt, fg_mask, tgt_label, ref_label):
        r"""Compute losses on the generated flow maps.

        Args:
            flow (tensor or list of tensors): Generated flow maps.
                warped_images (tensor or list of tensors): Warped images using the
                flow maps.
            tgt_image (tensor): Target image for the warped image.
                flow_gt (tensor or list of tensors): Ground truth flow maps.
            flow_conf_gt (tensor or list of tensors): Confidence for the ground
                truth flow maps.
            fg_mask (tensor): Foreground mask for the target image.
            tgt_label (tensor): Target label map.
            ref_label (tensor): Reference label map.
        Returns:
            (dict):
              - loss_flow_L1 (tensor): L1 loss compared to ground truth flow.
              - loss_flow_warp (tensor): L1 loss between the warped image and the
                target image when using the flow to warp.
              - body_mask_diff (tensor): Difference between warped body part map
                and target body part map. Used for pose dataset only.
        """
        loss_flow_L1 = torch.tensor(0., device=torch.device('cuda'))
        loss_flow_warp = torch.tensor(0., device=torch.device('cuda'))
        if isinstance(flow, list):
            # Compute flow losses for both warping reference -> target and
            # previous -> target.
            for i in range(len(flow)):
                loss_flow_L1_i, loss_flow_warp_i = \
                    self.compute_flow_loss(flow[i], warped_images[i], tgt_image,
                                           flow_gt[i], flow_conf_gt[i], fg_mask)
                loss_flow_L1 += loss_flow_L1_i
                loss_flow_warp += loss_flow_warp_i
        else:
            # Compute loss for warping either reference or previous images.
            loss_flow_L1, loss_flow_warp = \
                self.compute_flow_loss(flow, warped_images, tgt_image,
                                       flow_gt[-1], flow_conf_gt[-1], fg_mask)

        # For pose dataset only.
        body_mask_diff = None
        if self.warp_ref:
            if self.for_pose_dataset:
                # Warped reference body part map should be similar to target
                # body part map.
                body_mask = get_part_mask(tgt_label[:, 2])
                ref_body_mask = get_part_mask(ref_label[:, 2])
                warped_ref_body_mask = resample(ref_body_mask, flow[0])
                loss_flow_warp += self.criterion(warped_ref_body_mask,
                                                 body_mask)
                body_mask_diff = torch.sum(
                    abs(warped_ref_body_mask - body_mask), dim=1, keepdim=True)

            if self.has_fg:
                # Warped reference foreground map should be similar to target
                # foreground map.
                fg_mask, ref_fg_mask = \
                    get_fg_mask([tgt_label, ref_label], True)
                warped_ref_fg_mask = resample(ref_fg_mask, flow[0])
                loss_flow_warp += self.criterion(warped_ref_fg_mask, fg_mask)

        return loss_flow_L1, loss_flow_warp, body_mask_diff

    def compute_flow_loss(self, flow, warped_image, tgt_image, flow_gt,
                          flow_conf_gt, fg_mask):
        r"""Compute losses on the generated flow map.

        Args:
            flow (tensor): Generated flow map.
            warped_image (tensor): Warped image using the flow map.
            tgt_image (tensor): Target image for the warped image.
            flow_gt (tensor): Ground truth flow map.
            flow_conf_gt (tensor): Confidence for the ground truth flow map.
            fg_mask (tensor): Foreground mask for the target image.
        Returns:
            (dict):
              - loss_flow_L1 (tensor): L1 loss compared to ground truth flow.
              - loss_flow_warp (tensor): L1 loss between the warped image and
              the target image when using the flow to warp.
        """
        loss_flow_L1 = torch.tensor(0., device=torch.device('cuda'))
        loss_flow_warp = torch.tensor(0., device=torch.device('cuda'))
        if flow is not None and flow_gt is not None:
            # L1 loss compared to flow ground truth.
            loss_flow_L1 = self.criterionMasked(flow, flow_gt,
                                                flow_conf_gt * fg_mask)
        if warped_image is not None:
            # L1 loss between warped image and target image.
            loss_flow_warp = self.criterion(warped_image, tgt_image)
        return loss_flow_L1, loss_flow_warp

    def compute_mask_losses(self, occ_mask, fake_image, warped_image,
                            tgt_label, tgt_image, fg_mask, ref_fg_mask,
                            body_mask_diff):
        r"""Compute losses on the generated occlusion masks.

        Args:
            occ_mask (tensor or list of tensors): Generated occlusion masks.
            fake_image (tensor): Generated image.
            warped_image (tensor or list of tensors): Warped images using the
                flow maps.
            tgt_label (tensor): Target label map.
            tgt_image (tensor): Target image for the warped image.
            fg_mask (tensor): Foreground mask for the target image.
            ref_fg_mask (tensor): Foreground mask for the reference image.
            body_mask_diff (tensor): Difference between warped body part map
            and target body part map. Used for pose dataset only.
        Returns:
            (tensor): Loss for the mask.
        """
        loss_mask = torch.tensor(0., device=torch.device('cuda'))

        if isinstance(occ_mask, list):
            # Compute occlusion mask losses for both warping reference -> target
            # and previous -> target.
            for i in range(len(occ_mask)):
                loss_mask += self.compute_mask_loss(occ_mask[i],
                                                    warped_image[i],
                                                    tgt_image)
        else:
            # Compute loss for warping either reference or previous images.
            loss_mask += self.compute_mask_loss(occ_mask, warped_image,
                                                tgt_image)

        if self.warp_ref:
            ref_occ_mask = occ_mask[0]
            dummy0 = torch.zeros_like(ref_occ_mask)
            dummy1 = torch.ones_like(ref_occ_mask)
            if self.for_pose_dataset:
                # Enforce output to use more warped reference image for
                # face region.
                face_mask = get_face_mask(tgt_label[:, 2]).unsqueeze(1)
                AvgPool = torch.nn.AvgPool2d(15, padding=7, stride=1)
                face_mask = AvgPool(face_mask)
                loss_mask += self.criterionMasked(ref_occ_mask, dummy0,
                                                  face_mask)
                loss_mask += self.criterionMasked(fake_image, warped_image[0],
                                                  face_mask)
                # Enforce output to use more hallucinated image for discrepancy
                # regions of body part masks between warped reference and
                # target image.
                loss_mask += self.criterionMasked(ref_occ_mask, dummy1,
                                                  body_mask_diff)

            if self.has_fg:
                # Enforce output to use more hallucinated image for discrepancy
                # regions of foreground masks between reference and target
                # image.
                fg_mask_diff = ((ref_fg_mask - fg_mask) > 0).float()
                loss_mask += self.criterionMasked(ref_occ_mask, dummy1,
                                                  fg_mask_diff)
        return loss_mask

    def compute_mask_loss(self, occ_mask, warped_image, tgt_image):
        r"""Compute losses on the generated occlusion mask.

        Args:
            occ_mask (tensor): Generated occlusion mask.
            warped_image (tensor): Warped image using the flow map.
            tgt_image (tensor): Target image for the warped image.
        Returns:
            (tensor): Loss for the mask.
        """
        loss_mask = torch.tensor(0., device=torch.device('cuda'))
        if occ_mask is not None:
            dummy0 = torch.zeros_like(occ_mask)
            dummy1 = torch.ones_like(occ_mask)

            # Compute the confidence map based on L1 distance between warped
            # and GT image.
            img_diff = torch.sum(abs(warped_image - tgt_image), dim=1,
                                 keepdim=True)
            conf = torch.clamp(1 - img_diff, 0, 1)

            # Force mask value to be small if warped image is similar to GT,
            # and vice versa.
            loss_mask = self.criterionMasked(occ_mask, dummy0, conf)
            loss_mask += self.criterionMasked(occ_mask, dummy1, 1 - conf)

        return loss_mask
