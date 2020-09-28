# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""Utils for the few shot vid2vid model."""
import random

import numpy as np
import torch
import torch.nn.functional as F


def resample(image, flow):
    r"""Resamples an image using the provided flow.

    Args:
        image (NxCxHxW tensor) : Image to resample.
        flow (Nx2xHxW tensor) : Optical flow to resample the image.
    Returns:
        output (NxCxHxW tensor) : Resampled image.
    """
    assert flow.shape[1] == 2
    b, c, h, w = image.size()
    grid = get_grid(b, (h, w))
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    try:
        output = F.grid_sample(image, final_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
    except Exception:
        output = F.grid_sample(image, final_grid, mode='bilinear',
                               padding_mode='border')
    return output


def get_grid(batchsize, size, minval=-1.0, maxval=1.0):
    r"""Get a grid ranging [-1, 1] of 2D/3D coordinates.

    Args:
        batchsize (int) : Batch size.
        size (tuple) : (height, width) or (depth, height, width).
        minval (float) : minimum value in returned grid.
        maxval (float) : maximum value in returned grid.
    Returns:
        t_grid (4D tensor) : Grid of coordinates.
    """
    if len(size) == 2:
        rows, cols = size
    elif len(size) == 3:
        deps, rows, cols = size
    else:
        raise ValueError('Dimension can only be 2 or 3.')
    x = torch.linspace(minval, maxval, cols)
    x = x.view(1, 1, 1, cols)
    x = x.expand(batchsize, 1, rows, cols)

    y = torch.linspace(minval, maxval, rows)
    y = y.view(1, 1, rows, 1)
    y = y.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([x, y], dim=1)

    if len(size) == 3:
        z = torch.linspace(minval, maxval, deps)
        z = z.view(1, 1, deps, 1, 1)
        z = z.expand(batchsize, 1, deps, rows, cols)

        t_grid = t_grid.unsqueeze(2).expand(batchsize, 2, deps, rows, cols)
        t_grid = torch.cat([t_grid, z], dim=1)

    t_grid.requires_grad = False
    return t_grid.to('cuda')


def pick_image(images, idx):
    r"""Pick the image among images according to idx.

    Args:
        images (B x N x C x H x W tensor or list of tensors) : N images.
        idx (B tensor) : indices to select.
    Returns:
        image (B x C x H x W) : Selected images.
    """
    if type(images) == list:
        return [pick_image(r, idx) for r in images]
    if idx is None:
        return images[:, 0]
    elif type(idx) == int:
        return images[:, idx]
    idx = idx.long().view(-1, 1, 1, 1, 1)
    image = images.gather(1, idx.expand_as(images)[:, 0:1])[:, 0]
    return image


def crop_face_from_data(cfg, is_inference, data):
    r"""Crop the face regions in input data and resize to the target size.
    This is for training face datasets.

    Args:
        cfg (obj): Data configuration.
        is_inference (bool): Is doing inference or not.
        data (dict): Input data.
    Returns:
        data (dict): Cropped data.
    """
    label = data['label'] if 'label' in data else None
    image = data['images']
    landmarks = data['landmarks-dlib68_xy']
    ref_labels = data['few_shot_label'] if 'few_shot_label' in data else None
    ref_images = data['few_shot_images']
    ref_landmarks = data['few_shot_landmarks-dlib68_xy']
    img_size = image.shape[-2:]
    h, w = cfg.output_h_w.split(',')
    h, w = int(h), int(w)

    # When doing inference, need to sync common attributes like crop coodinates
    # between different workers, so all workers crop the same region.
    if 'common_attr' in data and 'crop_coords' in data['common_attr']:
        # Has been computed before, reusing the previous one.
        crop_coords, ref_crop_coords = data['common_attr']['crop_coords']
    else:
        # Is the first frame, need to compute the bbox.
        ref_crop_coords, scale = get_face_bbox_for_data(
            ref_landmarks[0], img_size, None, is_inference)
        crop_coords, _ = get_face_bbox_for_data(
            landmarks[0], img_size, scale, is_inference)

    # Crop the images according to the bbox and resize them to target size.
    label, image = crop_and_resize([label, image], crop_coords, (h, w))
    ref_labels, ref_images = crop_and_resize([ref_labels, ref_images],
                                             ref_crop_coords, (h, w))

    data['images'], data['few_shot_images'] = image, ref_images
    if label is not None:
        data['label'], data['few_shot_label'] = label, ref_labels
    if is_inference:
        if 'common_attr' not in data:
            data['common_attr'] = dict()
        data['common_attr']['crop_coords'] = crop_coords, ref_crop_coords
    return data


def get_face_bbox_for_data(keypoints, orig_img_size, scale, is_inference):
    r"""Get the bbox coordinates for face region.

    Args:
        keypoints (Nx2 tensor): Facial landmarks.
        orig_img_size (int tuple): Height and width of the input image size.
        scale (float): When training, randomly scale the crop size for
        augmentation.
        is_inference (bool): Is doing inference or not.
    Returns:
        crop_coords (list of int): bbox for face region.
        scale (float): Also returns scale to ensure reference and target frames
        are croppped using the same scale.
    """
    min_y, max_y = int(keypoints[:, 1].min()), int(keypoints[:, 1].max())
    min_x, max_x = int(keypoints[:, 0].min()), int(keypoints[:, 0].max())
    x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2
    H, W = orig_img_size
    w = h = (max_x - min_x)
    if not is_inference:
        # During training, randomly jitter the cropping position by offset
        # amount for augmentation.
        offset_max = 0.2
        offset = [np.random.uniform(-offset_max, offset_max),
                  np.random.uniform(-offset_max, offset_max)]
        # Also augment the crop size.
        if scale is None:
            scale_max = 0.2
            scale = [np.random.uniform(1 - scale_max, 1 + scale_max),
                     np.random.uniform(1 - scale_max, 1 + scale_max)]
        w *= scale[0]
        h *= scale[1]
        x_cen += int(offset[0] * w)
        y_cen += int(offset[1] * h)

    # Get the cropping coordinates.
    x_cen = max(w, min(W - w, x_cen))
    y_cen = max(h * 1.25, min(H - h * 0.75, y_cen))

    min_x = x_cen - w
    min_y = y_cen - h * 1.25
    max_x = min_x + w * 2
    max_y = min_y + h * 2

    crop_coords = [min_y, max_y, min_x, max_x]
    return [int(x) for x in crop_coords], scale


def crop_person_from_data(cfg, is_inference, data):
    r"""Crop the person regions in data and resize to the target size.
    This is for training full body datasets.

    Args:
        cfg (obj): Data configuration.
        is_inference (bool): Is doing inference or not.
        data (dict): Input data.
    Returns:
        data (dict): Cropped data.
    """
    label = data['label']
    image = data['images']
    use_few_shot = 'few_shot_label' in data
    if use_few_shot:
        ref_labels = data['few_shot_label']
        ref_images = data['few_shot_images']

    img_size = image.shape[-2:]
    output_h, output_w = cfg.output_h_w.split(',')
    output_h, output_w = int(output_h), int(output_w)
    output_aspect_ratio = output_w / output_h

    if 'human_instance_maps' in data:
        # Remove other people in the DensePose map except for the current
        # target.
        label = remove_other_ppl(label, data['human_instance_maps'])
        if use_few_shot:
            ref_labels = remove_other_ppl(ref_labels,
                                          data['few_shot_human_instance_maps'])

    # Randomly jitter the crop position by offset amount for augmentation.
    offset = ref_offset = None
    if not is_inference:
        offset = np.random.randn(2) * 0.05
        offset = np.minimum(1, np.maximum(-1, offset))
        ref_offset = np.random.randn(2) * 0.02
        ref_offset = np.minimum(1, np.maximum(-1, ref_offset))

    # Randomly scale the crop size for augmentation.
    # Final cropped size = person height * scale.
    scale = ref_scale = 1.5
    if not is_inference:
        scale = min(2, max(1, scale + np.random.randn() * 0.05))
        ref_scale = min(2, max(1, ref_scale + np.random.randn() * 0.02))

    # When doing inference, need to sync common attributes like crop coodinates
    # between different workers, so all workers crop the same region.
    if 'common_attr' in data:
        # Has been computed before, reusing the previous one.
        crop_coords, ref_crop_coords = data['common_attr']['crop_coords']
    else:
        # Is the first frame, need to compute the bbox.
        crop_coords = get_person_bbox_for_data(label, img_size, scale,
                                               output_aspect_ratio, offset)
        if use_few_shot:
            ref_crop_coords = get_person_bbox_for_data(
                ref_labels, img_size, ref_scale,
                output_aspect_ratio, ref_offset)
        else:
            ref_crop_coords = None

    # Crop the images according to the bbox and resize them to target size.
    label = crop_and_resize(label, crop_coords, (output_h, output_w), 'nearest')
    image = crop_and_resize(image, crop_coords, (output_h, output_w))
    if use_few_shot:
        ref_labels = crop_and_resize(ref_labels, ref_crop_coords,
                                     (output_h, output_w), 'nearest')
        ref_images = crop_and_resize(ref_images, ref_crop_coords,
                                     (output_h, output_w))

    data['label'], data['images'] = label, image
    if use_few_shot:
        data['few_shot_label'], data['few_shot_images'] = ref_labels, ref_images
    if 'human_instance_maps' in data:
        del data['human_instance_maps']
    if 'few_shot_human_instance_maps' in data:
        del data['few_shot_human_instance_maps']
    if is_inference:
        data['common_attr'] = dict()
        data['common_attr']['crop_coords'] = crop_coords, ref_crop_coords

    return data


def get_person_bbox_for_data(pose_map, orig_img_size, scale=1.5,
                             crop_aspect_ratio=1, offset=None):
    r"""Get the bbox (pixel coordinates) to crop for person body region.

    Args:
        pose_map (NxCxHxW tensor): Input pose map.
        orig_img_size (int tuple): Height and width of the input image size.
        scale (float): When training, randomly scale the crop size for
        augmentation.
        crop_aspect_ratio (float): Output aspect ratio,
        offset (list of float): Offset for crop position.
    Returns:
        crop_coords (list of int): bbox for body region.
    """
    H, W = orig_img_size
    assert pose_map.dim() == 4
    nonzero_indices = (pose_map[:, :3] > 0).nonzero(as_tuple=False)
    if nonzero_indices.size(0) == 0:
        bw = int(H * crop_aspect_ratio // 2)
        return [0, H, W // 2 - bw, W // 2 + bw]

    y_indices, x_indices = nonzero_indices[:, 2], nonzero_indices[:, 3]
    y_min, y_max = y_indices.min().item(), y_indices.max().item()
    x_min, x_max = x_indices.min().item(), x_indices.max().item()
    y_cen = int(y_min + y_max) // 2
    x_cen = int(x_min + x_max) // 2
    y_len = y_max - y_min
    x_len = x_max - x_min

    # bh, bw: half of height / width of final cropped size.
    bh = int(min(H, max(H // 2, y_len * scale))) // 2
    bh = max(bh, int(x_len * scale / crop_aspect_ratio) // 2)
    bw = int(bh * crop_aspect_ratio)

    # Randomly offset the cropped position for augmentation.
    if offset is not None:
        x_cen += int(offset[0] * bw)
        y_cen += int(offset[1] * bh)
    x_cen = max(bw, min(W - bw, x_cen))
    y_cen = max(bh, min(H - bh, y_cen))

    return [(y_cen - bh), (y_cen + bh), (x_cen - bw), (x_cen + bw)]


def crop_and_resize(img, coords, size=None, method='bilinear'):
    r"""Crop the image using the given coordinates and resize to target size.

    Args:
        img (tensor or list of tensors): Input image.
        coords (list of int): Pixel coordinates to crop.
        size (list of int): Output size.
        method (str): Interpolation method.
    Returns:
        img (tensor or list of tensors): Output image.
    """
    if isinstance(img, list):
        return [crop_and_resize(x, coords, size, method) for x in img]
    if img is None:
        return None
    min_y, max_y, min_x, max_x = coords

    img = img[:, :, min_y:max_y, min_x:max_x]
    if size is not None:
        if method == 'nearest':
            img = F.interpolate(img, size=size, mode=method)
        else:
            img = F.interpolate(img, size=size, mode=method,
                                align_corners=False)
    return img


def remove_other_ppl(labels, densemasks):
    r"""Remove other people in the label map except for the current target
    by looking at the id in the densemask map.

    Args:
        labels (NxCxHxW tensor): Input labels.
        densemasks (Nx1xHxW tensor): Densemask maps.
    Returns:
        labels (NxCxHxW tensor): Output labels.
    """
    densemasks = densemasks[:, 0:1] * 255
    for idx in range(labels.shape[0]):
        label, densemask = labels[idx], densemasks[idx]
        # Get OpenPose and find the person id in Densemask that has the most
        # overlap with the person in OpenPose result.
        openpose = label[3:]
        valid = (openpose[0] > 0) | (openpose[1] > 0) | (openpose[2] > 0)
        dp_valid = densemask[valid.unsqueeze(0)]
        if dp_valid.shape[0]:
            ind = np.bincount(dp_valid).argmax()
            # Remove all other people that have different indices.
            label = label * (densemask == ind).float()
        labels[idx] = label
    return labels


def select_object(data, obj_indices=None):
    r"""Select the object/person in the dict according to the object index.
    Currently it's used to select the target person in OpenPose dict.

    Args:
        data (dict): Input data.
        obj_indices (list of int): Indices for the objects to select.
    Returns:
        data (dict): Output data.
    """
    op_key = 'poses-openpose'
    if op_key in data:
        for i in range(len(data[op_key])):
            # data[op_key] is a list of dicts for different frames.
            # people = data[op_key][i]['people']
            people = data[op_key][i]
            # "people" is a list of people dicts found by OpenPose. We will
            # use the obj_index to get the target person from the list, and
            # write it back to the dict.
            # data[op_key][i]['people'] = [people[obj_indices[i]]]
            if obj_indices is not None:
                data[op_key][i] = people[obj_indices[i]]
            else:
                data[op_key][i] = people[0]
    return data


def concat_frames(prev, now, n_frames):
    r"""Concat previous and current frames and only keep the latest $(n_frames).
    If concatenated frames are longer than $(n_frames), drop the oldest one.

    Args:
        prev (NxTxCxHxW tensor): Tensor for previous frames.
        now (NxCxHxW tensor): Tensor for current frame.
        n_frames (int): Max number of frames to store.
    Returns:
        result (NxTxCxHxW tensor): Updated tensor.
    """
    now = now.unsqueeze(1)
    if prev is None:
        return now
    if prev.shape[1] == n_frames:
        prev = prev[:, 1:]
    return torch.cat([prev, now], dim=1)


def combine_fg_mask(fg_mask, ref_fg_mask, has_fg):
    r"""Get the union of target and reference foreground masks.
    Args:
        fg_mask (tensor): Foreground mask for target image.
        ref_fg_mask (tensor): Foreground mask for reference image.
        has_fg (bool): Whether the image can be classified into fg/bg.
    Returns:
        output (tensor or int): Combined foreground mask.
    """
    return ((fg_mask > 0) | (ref_fg_mask > 0)).float() if has_fg else 1


def get_fg_mask(densepose_map, has_fg):
    r"""Obtain the foreground mask for pose sequences, which only includes
    the human. This is done by looking at the body part map from DensePose.

    Args:
        densepose_map (NxCxHxW tensor): DensePose map.
        has_fg (bool): Whether data has foreground or not.
    Returns:
        mask (Nx1xHxW tensor): fg mask.
    """
    if type(densepose_map) == list:
        return [get_fg_mask(label, has_fg) for label in densepose_map]
    if not has_fg or densepose_map is None:
        return 1
    if densepose_map.dim() == 5:
        densepose_map = densepose_map[:, 0]
    # Get the body part map from DensePose.
    mask = densepose_map[:, 2:3]

    # Make the mask slightly larger.
    mask = torch.nn.MaxPool2d(15, padding=7, stride=1)(mask)
    mask = (mask > -1).float()
    return mask


def get_part_mask(densepose_map):
    r"""Obtain mask of different body parts of humans. This is done by
    looking at the body part map from DensePose.

    Args:
        densepose_map (NxCxHxW tensor): DensePose map.
    Returns:
        mask (NxKxHxW tensor): Body part mask, where K is the number of parts.
    """
    # Groups of body parts. Each group contains IDs of body part labels in
    # DensePose. The 9 groups here are: background, torso, hands, feet,
    # upper legs, lower legs, upper arms, lower arms, head.
    part_groups = [[0], [1, 2], [3, 4], [5, 6], [7, 9, 8, 10], [11, 13, 12, 14],
                   [15, 17, 16, 18], [19, 21, 20, 22], [23, 24]]
    n_parts = len(part_groups)

    need_reshape = densepose_map.dim() == 4
    if need_reshape:
        bo, t, h, w = densepose_map.size()
        densepose_map = densepose_map.view(-1, h, w)
    b, h, w = densepose_map.size()
    part_map = (densepose_map / 2 + 0.5) * 24
    assert (part_map >= 0).all() and (part_map < 25).all()

    mask = torch.cuda.ByteTensor(b, n_parts, h, w).fill_(0)
    for i in range(n_parts):
        for j in part_groups[i]:
            # Account for numerical errors.
            mask[:, i] = mask[:, i] | (
                (part_map > j - 0.1) & (part_map < j + 0.1)).byte()
    if need_reshape:
        mask = mask.view(bo, t, -1, h, w)
    return mask.float()


def get_face_mask(densepose_map):
    r"""Obtain mask of faces.
    Args:
        densepose_map (3D or 4D tensor): DensePose map.
    Returns:
        mask (3D or 4D tensor): Face mask.
    """
    need_reshape = densepose_map.dim() == 4
    if need_reshape:
        bo, t, h, w = densepose_map.size()
        densepose_map = densepose_map.view(-1, h, w)

    b, h, w = densepose_map.size()
    part_map = (densepose_map / 2 + 0.5) * 24
    assert (part_map >= 0).all() and (part_map < 25).all()
    if densepose_map.is_cuda:
        mask = torch.cuda.ByteTensor(b, h, w).fill_(0)
    else:
        mask = torch.ByteTensor(b, h, w).fill_(0)
    for j in [23, 24]:
        mask = mask | ((part_map > j - 0.1) & (part_map < j + 0.1)).byte()
    if need_reshape:
        mask = mask.view(bo, t, h, w)
    return mask.float()


def extract_valid_pose_labels(pose_map, pose_type, remove_face_labels,
                              do_remove=True):
    r"""Remove some labels (e.g. face regions) in the pose map if necessary.

    Args:
        pose_map (3D, 4D or 5D tensor): Input pose map.
        pose_type (str): 'both' or 'open'.
        remove_face_labels (bool): Whether to remove labels for the face region.
        do_remove (bool): Do remove face labels.
    Returns:
        pose_map (3D, 4D or 5D tensor): Output pose map.
    """
    if pose_map is None:
        return pose_map
    if type(pose_map) == list:
        return [extract_valid_pose_labels(p, pose_type, remove_face_labels,
                                          do_remove) for p in pose_map]

    orig_dim = pose_map.dim()
    assert (orig_dim >= 3 and orig_dim <= 5)
    if orig_dim == 3:
        pose_map = pose_map.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 4:
        pose_map = pose_map.unsqueeze(0)

    if pose_type == 'open':
        # If input is only openpose, remove densepose part.
        pose_map = pose_map[:, :, 3:]

    elif remove_face_labels and do_remove:
        # Remove face part for densepose input.
        densepose, openpose = pose_map[:, :, :3], pose_map[:, :, 3:]
        face_mask = get_face_mask(pose_map[:, :, 2]).unsqueeze(2)
        pose_map = torch.cat([densepose * (1 - face_mask) - face_mask,
                              openpose], dim=2)

    if orig_dim == 3:
        pose_map = pose_map[0, 0]
    elif orig_dim == 4:
        pose_map = pose_map[0]
    return pose_map


def normalize_faces(keypoints, ref_keypoints,
                    dist_scale_x=None, dist_scale_y=None):
    r"""Normalize face keypoints w.r.t. the reference face keypoints.

    Args:
        keypoints (Kx2 numpy array): target facial keypoints.
        ref_keypoints (Kx2 numpy array): reference facial keypoints.
    Returns:
        keypoints (Kx2 numpy array): normalized facial keypoints.
    """
    if keypoints.shape[0] == 68:
        central_keypoints = [8]
        add_upper_face = False
        part_list = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12],
                     [5, 11], [6, 10], [7, 9, 8],
                     [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
                     [27], [28], [29], [30], [31, 35], [32, 34], [33],
                     [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                     [48, 54], [49, 53], [50, 52], [51], [55, 59], [56, 58],
                     [57],
                     [60, 64], [61, 63], [62], [65, 67], [66]
                     ]
        if add_upper_face:
            part_list += [[68, 82], [69, 81], [70, 80], [71, 79], [72, 78],
                          [73, 77], [74, 76, 75]]
    elif keypoints.shape[0] == 126:
        central_keypoints = [16]
        part_list = [[i] for i in range(126)]
    else:
        raise ValueError('Input keypoints type not supported.')

    face_cen = np.mean(keypoints[central_keypoints, :], axis=0)
    ref_face_cen = np.mean(ref_keypoints[central_keypoints, :], axis=0)

    def get_mean_dists(pts, face_cen):
        r"""Get the mean xy distances of keypoints wrt face center."""
        mean_dists_x, mean_dists_y = [], []
        pts_cen = np.mean(pts, axis=0)
        for p, pt in enumerate(pts):
            mean_dists_x.append(np.linalg.norm(pt - pts_cen))
            mean_dists_y.append(np.linalg.norm(pts_cen - face_cen))
        mean_dist_x = sum(mean_dists_x) / len(mean_dists_x) + 1e-3
        mean_dist_y = sum(mean_dists_y) / len(mean_dists_y) + 1e-3
        return mean_dist_x, mean_dist_y

    if dist_scale_x is None:
        dist_scale_x, dist_scale_y = [None] * len(part_list), \
                                     [None] * len(part_list)

    for i, pts_idx in enumerate(part_list):
        pts = keypoints[pts_idx]
        if dist_scale_x[i] is None:
            ref_pts = ref_keypoints[pts_idx]
            mean_dist_x, mean_dist_y = get_mean_dists(pts, face_cen)
            ref_dist_x, ref_dist_y = get_mean_dists(ref_pts, ref_face_cen)

            dist_scale_x[i] = ref_dist_x / mean_dist_x
            dist_scale_y[i] = ref_dist_y / mean_dist_y

        pts_cen = np.mean(pts, axis=0)
        pts = (pts - pts_cen) * dist_scale_x[i] + \
              (pts_cen - face_cen) * dist_scale_y[i] + face_cen
        keypoints[pts_idx] = pts
    return keypoints, [dist_scale_x, dist_scale_y]


def crop_face_from_output(data_cfg, image, input_label, crop_smaller=0):
    r"""Crop out the face region of the image (and resize if necessary to feed
    into generator/discriminator).

    Args:
        data_cfg (obj): Data configuration.
        image (NxC1xHxW tensor or list of tensors): Image to crop.
        input_label (NxC2xHxW tensor): Input label map.
        crop_smaller (int): Number of pixels to crop slightly smaller region.
    Returns:
        output (NxC1xHxW tensor or list of tensors): Cropped image.
    """
    if type(image) == list:
        return [crop_face_from_output(data_cfg, im, input_label, crop_smaller)
                for im in image]

    output = None
    face_size = image.shape[-2] // 32 * 8
    for i in range(input_label.size(0)):
        ys, ye, xs, xe = get_face_bbox_for_output(data_cfg,
                                                  input_label[i:i + 1],
                                                  crop_smaller=crop_smaller)
        output_i = F.interpolate(image[i:i + 1, -3:, ys:ye, xs:xe],
                                 size=(face_size, face_size), mode='bilinear',
                                 align_corners=True)
        # output_i = image[i:i + 1, -3:, ys:ye, xs:xe]
        output = torch.cat([output, output_i]) if i != 0 else output_i
    return output


def get_face_bbox_for_output(data_cfg, pose, crop_smaller=0):
    r"""Get pixel coordinates of the face bounding box.

    Args:
        data_cfg (obj): Data configuration.
        pose (NxCxHxW tensor): Pose label map.
        crop_smaller (int): Number of pixels to crop slightly smaller region.
    Returns:
        output (list of int): Face bbox.
    """
    if pose.dim() == 3:
        pose = pose.unsqueeze(0)
    elif pose.dim() == 5:
        pose = pose[-1, -1:]
    _, _, h, w = pose.size()

    use_openpose = 'pose_maps-densepose' not in data_cfg.input_labels
    if use_openpose:  # Use openpose face keypoints to identify face region.
        for input_type in data_cfg.input_types:
            if 'poses-openpose' in input_type:
                num_ch = input_type['poses-openpose'].num_channels
        if num_ch > 3:
            face = (pose[:, -1] > 0).nonzero(as_tuple=False)
        else:
            raise ValueError('Not implemented yet.')
    else:  # Use densepose labels.
        face = (pose[:, 2] > 0.9).nonzero(as_tuple=False)

    ylen = xlen = h // 32 * 8
    if face.size(0):
        y, x = face[:, 1], face[:, 2]
        ys, ye = y.min().item(), y.max().item()
        xs, xe = x.min().item(), x.max().item()
        if use_openpose:
            xc, yc = (xs + xe) // 2, (ys * 3 + ye * 2) // 5
            ylen = int((xe - xs) * 2.5)
        else:
            xc, yc = (xs + xe) // 2, (ys + ye) // 2
            ylen = int((ye - ys) * 1.25)
        ylen = xlen = min(w, max(32, ylen))
        yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
        xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))
    else:
        yc = h // 4
        xc = w // 2

    ys, ye = yc - ylen // 2, yc + ylen // 2
    xs, xe = xc - xlen // 2, xc + xlen // 2
    if crop_smaller != 0:  # Crop slightly smaller region inside face.
        ys += crop_smaller
        xs += crop_smaller
        ye -= crop_smaller
        xe -= crop_smaller
    return [ys, ye, xs, xe]


def crop_hand_from_output(data_cfg, image, input_label):
    r"""Crop out the hand region of the image.

    Args:
        data_cfg (obj): Data configuration.
        image (NxC1xHxW tensor or list of tensors): Image to crop.
        input_label (NxC2xHxW tensor): Input label map.
    Returns:
        output (NxC1xHxW tensor or list of tensors): Cropped image.
    """
    if type(image) == list:
        return [crop_hand_from_output(data_cfg, im, input_label)
                for im in image]

    output = None
    for i in range(input_label.size(0)):
        coords = get_hand_bbox_for_output(data_cfg, input_label[i:i + 1])
        if coords:
            for coord in coords:
                ys, ye, xs, xe = coord
                output_i = image[i:i + 1, -3:, ys:ye, xs:xe]
                output = torch.cat([output, output_i]) \
                    if output is not None else output_i
    return output


def get_hand_bbox_for_output(data_cfg, pose):
    r"""Get coordinates of the hand bounding box.

    Args:
        data_cfg (obj): Data configuration.
        pose (NxCxHxW tensor): Pose label map.
    Returns:
        output (list of int): Hand bbox.
    """
    if pose.dim() == 3:
        pose = pose.unsqueeze(0)
    elif pose.dim() == 5:
        pose = pose[-1, -1:]
    _, _, h, w = pose.size()
    ylen = xlen = h // 64 * 8

    coords = []
    colors = [[0.95, 0.5, 0.95], [0.95, 0.95, 0.5]]
    for i, color in enumerate(colors):
        if pose.shape[1] > 6:  # Using one-hot encoding for openpose.
            idx = -3 if i == 0 else -2
            hand = (pose[:, idx] == 1).nonzero(as_tuple=False)
        else:
            raise ValueError('Not implemented yet.')
        if hand.size(0):
            y, x = hand[:, 1], hand[:, 2]
            ys, ye, xs, xe = y.min().item(), y.max().item(), \
                x.min().item(), x.max().item()
            xc, yc = (xs + xe) // 2, (ys + ye) // 2
            yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
            xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))
            ys, ye, xs, xe = yc - ylen // 2, yc + ylen // 2, \
                xc - xlen // 2, xc + xlen // 2
            coords.append([ys, ye, xs, xe])
    return coords


def pre_process_densepose(pose_cfg, pose_map, is_infer=False):
    r"""Pre-process the DensePose part of input label map.

    Args:
        pose_cfg (obj): Pose data configuration.
        pose_map (NxCxHxW tensor): Pose label map.
        is_infer (bool): Is doing inference.
    Returns:
        pose_map (NxCxHxW tensor): Processed pose label map.
    """
    part_map = pose_map[:, :, 2] * 255  # should be within [0-24]
    assert (part_map >= 0).all() and (part_map < 25).all()

    # Randomly drop some body part during training.
    if not is_infer:
        random_drop_prob = getattr(pose_cfg, 'random_drop_prob', 0)
    else:
        random_drop_prob = 0
    if random_drop_prob > 0:
        densepose_map = pose_map[:, :, :3]
        for part_id in range(1, 25):
            if (random.random() < random_drop_prob):
                part_mask = abs(part_map - part_id) < 0.1
                densepose_map[part_mask.unsqueeze(2).expand_as(
                    densepose_map)] = 0
        pose_map[:, :, :3] = densepose_map

    # Renormalize the DensePose channel from [0, 24] to [0, 255].
    pose_map[:, :, 2] = pose_map[:, :, 2] * (255 / 24)
    # Normalize from [0, 1] to [-1, 1].
    pose_map = pose_map * 2 - 1
    return pose_map


def random_roll(tensors):
    r"""Randomly roll the input tensors along x and y dimensions. Also randomly
    flip the tensors.

    Args:
        tensors (list of 4D tensors): Input tensors.
    Returns:
        output (list of 4D tensors): Rolled tensors.
    """
    h, w = tensors[0].shape[2:]
    ny = np.random.choice([np.random.randint(h//16),
                           h-np.random.randint(h//16)])
    nx = np.random.choice([np.random.randint(w//16),
                           w-np.random.randint(w//16)])
    flip = np.random.rand() > 0.5
    return [roll(t, ny, nx, flip) for t in tensors]


def roll(t, ny, nx, flip):
    r"""Roll and flip the tensor by specified amounts.

    Args:
        t (4D tensor): Input tensor.
        ny (int): Amount to roll along y dimension.
        nx (int): Amount to roll along x dimension.
        flip (bool): Whether to flip input.
    Returns:
        t (4D tensor): Output tensor.
    """
    t = torch.cat([t[:, :, -ny:], t[:, :, :-ny]], dim=2)
    t = torch.cat([t[:, :, :, -nx:], t[:, :, :, :-nx]], dim=3)
    if flip:
        t = torch.flip(t, dims=[3])
    return t


def detach(output):
    r"""Detach tensors in the dict.

    Args:
        output (dict): Output dict.
    Returns:
        output (dict): Detached output dict.
    """
    if type(output) == dict:
        new_dict = dict()
        for k, v in output.items():
            new_dict[k] = detach(v)
        return new_dict
    elif type(output) == torch.Tensor:
        return output.detach()
    return output
