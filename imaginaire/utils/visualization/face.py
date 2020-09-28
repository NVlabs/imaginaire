# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import numpy as np
import cv2
import torch
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import warnings
from imaginaire.utils.io import get_checkpoint


def connect_face_keypoints(resize_h, resize_w, crop_h, crop_w, original_h,
                           original_w, is_flipped, cfgdata, keypoints):
    r"""Connect the face keypoints to edges and draw the sketch.

    Args:
        resize_h (int): Height the input image was resized to.
        resize_w (int): Width the input image was resized to.
        crop_h (int): Height the input image was cropped.
        crop_w (int): Width the input image was cropped.
        original_h (int): Original height of the input image.
        original_w (int): Original width of the input image.
        is_flipped (bool): Is the input image flipped.
        cfgdata (obj): Data configuration.
        keypoints (NxKx2 numpy array): Facial landmarks (with K keypoints).

    Returns:
        (list of HxWxC numpy array): Drawn label map.
    """
    if hasattr(cfgdata, 'for_face_dataset'):
        face_cfg = cfgdata.for_face_dataset
        # Whether to add the upper part of face to label map.
        add_upper_face = getattr(face_cfg, 'add_upper_face', False)
        # Whether to add distance transform output to label map.
        add_dist_map = getattr(face_cfg, 'add_distance_transform', False)
        # Whether to add positional encoding to label map.
        add_pos_encode = add_dist_map and getattr(
            face_cfg, 'add_positional_encode', False)
    else:
        add_upper_face = add_dist_map = add_pos_encode = False

    # Mapping from keypoint index to facial part.
    part_list = [[list(range(0, 17)) + (
        (list(range(68, 83)) + [0]) if add_upper_face else [])],  # ai_emoji
                      [range(17, 22)],  # right eyebrow
                      [range(22, 27)],  # left eyebrow
                      [[28, 31], range(31, 36), [35, 28]],  # nose
                      [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                      [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                      [range(48, 55), [54, 55, 56, 57, 58, 59, 48],
                       range(60, 65), [64, 65, 66, 67, 60]],  # mouth and tongue
    ]
    if add_upper_face:
        pts = keypoints[:, :17, :].astype(np.int32)
        baseline_y = (pts[:, 0:1, 1] + pts[:, -1:, 1]) / 2
        upper_pts = pts[:, 1:-1, :].copy()
        upper_pts[:, :, 1] = baseline_y + (
                baseline_y - upper_pts[:, :, 1]) * 2 // 3
        keypoints = np.hstack((keypoints, upper_pts[:, ::-1, :]))

    edge_len = 3  # Interpolate 3 keypoints to form a curve when drawing edges.
    bw = max(1, resize_h // 256)  # Width of the stroke.

    outputs = []
    for t_idx in range(keypoints.shape[0]):
        # Edge map for the face region from keypoints.
        im_edges = np.zeros((resize_h, resize_w, 1), np.uint8)
        im_dists = np.zeros((resize_h, resize_w, 0), np.uint8)
        for edge_list in part_list:
            for e, edge in enumerate(edge_list):
                # Edge map for the current edge.
                im_edge = np.zeros((resize_h, resize_w, 1), np.uint8)
                # Divide a long edge into multiple small edges when drawing.
                for i in range(0, max(1, len(edge) - 1), edge_len - 1):
                    sub_edge = edge[i:i + edge_len]
                    x = keypoints[t_idx, sub_edge, 0]
                    y = keypoints[t_idx, sub_edge, 1]

                    # Interp keypoints to get the curve shape.
                    curve_x, curve_y = interp_points(x, y)
                    draw_edge(im_edges, curve_x, curve_y, bw=bw)
                    if add_dist_map:
                        draw_edge(im_edge, curve_x, curve_y, bw=bw)

                if add_dist_map:
                    # Add distance transform map on each facial part.
                    im_dist = cv2.distanceTransform(255 - im_edge,
                                                    cv2.DIST_L1, 3)
                    im_dist = np.clip((im_dist / 3), 0, 255)
                    im_dists = np.dstack((im_dists, im_dist))

                if add_pos_encode and e == 0:
                    # Add positional encoding for the first edge.
                    from math import pi
                    im_pos = np.zeros((resize_h, resize_w, 0), np.float32)
                    for l in range(10):  # noqa: E741
                        dist = (im_dist.astype(np.float32) - 127.5) / 127.5
                        sin = np.sin(pi * (2 ** l) * dist)
                        cos = np.cos(pi * (2 ** l) * dist)
                        im_pos = np.dstack((im_pos, sin, cos))

        # Combine all components to form the final label map.
        if add_dist_map:
            im_edges = np.dstack((im_edges, im_dists))
        im_edges = im_edges.astype(np.float32) / 255.0
        if add_pos_encode:
            im_edges = np.dstack((im_edges, im_pos))
        outputs.append(im_edges)
    return outputs


def normalize_and_connect_face_keypoints(cfg, is_inference, data):
    r"""Normalize face keypoints w.r.t. reference face keypoints and connect
    keypoints to form 2D images.

    Args:
        cfg (obj): Data configuration.
        is_inference (bool): Is doing inference or not.
        data (dict): Input data.

    Returns:
        (dict): Output data.
    """
    assert is_inference
    resize_h, resize_w = data['images'][0].shape[-2:]

    keypoints = data['label'].numpy()[0]
    ref_keypoints = data['few_shot_label'].numpy()[0]

    # Get the normalization params and prev data if it's been computed before.
    dist_scales = prev_keypoints = None
    if 'common_attr' in data and 'prev_data' in data['common_attr']:
        dist_scales = data['common_attr']['dist_scales']
        prev_keypoints = data['common_attr']['prev_data']

    def concat(prev, now, t):
        r"""Concat prev and now frames in first dimension, up to t frames."""
        if prev is None:
            return now
        return np.vstack([prev, now])[-t:]

    # Normalize face keypoints w.r.t. reference face keypoints.
    keypoints, dist_scales = \
        normalize_face_keypoints(keypoints[0], ref_keypoints[0], dist_scales,
                                 momentum=getattr(cfg.for_face_dataset,
                                                  'normalize_momentum', 0.9))
    keypoints = keypoints[np.newaxis, :]

    # Temporally smooth the face keypoints by median filtering.
    ks = getattr(cfg.for_face_dataset, 'smooth_kernel_size', 5)
    concat_keypoints = concat(prev_keypoints, keypoints, ks)
    if ks > 1 and concat_keypoints.shape[0] == ks:
        keypoints = smooth_face_keypoints(concat_keypoints, ks)

    # Store the computed params.
    if 'common_attr' not in data:
        data['common_attr'] = dict()
    data['common_attr']['dist_scales'] = dist_scales
    data['common_attr']['prev_data'] = concat_keypoints

    # Draw the keypoints to turn them into images.
    labels = []
    for kpt in [keypoints, ref_keypoints]:
        label = connect_face_keypoints(resize_h, resize_w, None, None, None,
                                       None, False, cfg, kpt)
        labels += [torch.from_numpy(label[0]).permute(2, 0, 1).unsqueeze(0)]
    data['label'], data['few_shot_label'] = labels
    return data


def smooth_face_keypoints(concat_keypoints, ks):
    r""" Temporally smooth the face keypoints by median filtering.

    Args:
        concat_keypoints (TxKx2 numpy array): Face keypoints to be filtered.
        ks (int): Filter kernel size.

    Returns:
        (1xKx2 numpy array): Output face keypoints.
    """
    # Median filtering.
    filtered_keypoints = medfilt(concat_keypoints, kernel_size=[ks, 1, 1])
    # Fill in any zero keypoints with the value from previous frame.
    if (filtered_keypoints == 0).any():
        for t in range(1, filtered_keypoints.shape[0]):
            kpt_prev = filtered_keypoints[t - 1]
            kpt_cur = filtered_keypoints[t]
            kpt_max = np.maximum(kpt_cur, kpt_prev)
            kpt_cur[kpt_cur == 0] = kpt_max[kpt_cur == 0]
            filtered_keypoints[t] = kpt_cur
    keypoints = filtered_keypoints[ks // 2: ks // 2 + 1]
    return keypoints


def normalize_face_keypoints(keypoints, ref_keypoints, dist_scales=None,
                             momentum=0.9):
    r"""Normalize face keypoints w.r.t. the reference face keypoints.

    Args:
        keypoints (Kx2 numpy array): Target facial keypoints to be normalized.
        ref_keypoints (Kx2 numpy array): Reference facial keypoints.
        dist_scales (list of list of floats): Normalization params.
        momentum (float): Temporal momentum for the normalization params.

    Returns:
        (Kx2 numpy array): Normalized facial keypoints.
    """
    if keypoints.shape[0] == 68:
        central_keypoints = [8]
        part_list = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12],
                     [5, 11], [6, 10], [7, 9, 8],
                     [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
                     [27], [28], [29], [30], [31, 35], [32, 34], [33],
                     [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                     [48, 54], [49, 53], [50, 52], [51], [55, 59], [56, 58],
                     [57],
                     [60, 64], [61, 63], [62], [65, 67], [66]
                     ]
    else:
        raise ValueError('Input keypoints type not supported.')

    face_cen = np.mean(keypoints[central_keypoints, :], axis=0)
    ref_face_cen = np.mean(ref_keypoints[central_keypoints, :], axis=0)

    def get_mean_dists(pts, face_cen):
        r"""Get mean distances of the points from face center."""
        mean_dists_x, mean_dists_y = [], []
        pts_cen = np.mean(pts, axis=0)
        for p, pt in enumerate(pts):
            mean_dists_x.append(np.linalg.norm(pt - pts_cen))
            mean_dists_y.append(np.linalg.norm(pts_cen - face_cen))
        mean_dist_x = sum(mean_dists_x) / len(mean_dists_x) + 1e-3
        mean_dist_y = sum(mean_dists_y) / len(mean_dists_y) + 1e-3
        return mean_dist_x, mean_dist_y

    dist_scale_x, dist_scale_y = [None] * len(part_list), \
                                 [None] * len(part_list)
    if dist_scales is None:
        dist_scale_x_prev = dist_scale_y_prev = img_scale = None
    else:
        dist_scale_x_prev, dist_scale_y_prev, img_scale = dist_scales
    if img_scale is None:
        img_scale = (keypoints[:, 0].max() - keypoints[:, 0].min()) \
                    / (ref_keypoints[:, 0].max() - ref_keypoints[:, 0].min())

    for i, pts_idx in enumerate(part_list):
        pts = keypoints[pts_idx]
        pts = pts[pts[:, 0] != 0]
        if pts.shape[0]:
            ref_pts = ref_keypoints[pts_idx]
            mean_dist_x, mean_dist_y = get_mean_dists(pts, face_cen)
            ref_dist_x, ref_dist_y = get_mean_dists(ref_pts, ref_face_cen)
            dist_scale_x[i] = ref_dist_x / mean_dist_x * img_scale
            dist_scale_y[i] = ref_dist_y / mean_dist_y * img_scale
            if dist_scale_x_prev is not None:
                dist_scale_x[i] = dist_scale_x_prev[i] * momentum + \
                    dist_scale_x[i] * (1 - momentum)
                dist_scale_y[i] = dist_scale_y_prev[i] * momentum + \
                    dist_scale_y[i] * (1 - momentum)

            pts_cen = np.mean(pts, axis=0)
            pts = (pts - pts_cen) * dist_scale_x[i] + \
                  (pts_cen - face_cen) * dist_scale_y[i] + face_cen
            keypoints[pts_idx] = pts

    return keypoints, [dist_scale_x, dist_scale_y, img_scale]


def npy_to_tensor(keypoints):
    r"""Convert numpy array to pytorch tensor."""
    return torch.from_numpy(keypoints).unsqueeze(0)


def get_dlib_landmarks_from_image(
        imgs, predictor_path='shape_predictor_68_face_landmarks.dat'):
    r"""Get face keypoints from an image.

    Args:
        imgs (N x 3 x H x W tensor or N x H x W x 3 numpy array): Input images.
        predictor_path (str): Path to the predictor model.
    """
    import dlib
    predictor_path = get_checkpoint(predictor_path,
                                    url='1l9zT-AI1yKlfyAb_wl_RjLBSaiWQr8dr')
    if type(imgs) == torch.Tensor:
        imgs = ((imgs + 1) / 2 * 255).byte()
        imgs = np.transpose(imgs.cpu().numpy(), (0, 2, 3, 1))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    points = np.zeros([imgs.shape[0], 68, 2], dtype=int)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        dets = detector(img, 1)
        if len(dets) > 0:
            # Only returns the first face.
            shape = predictor(img, dets[0])
            for b in range(68):
                points[i, b, 0] = shape.part(b).x
                points[i, b, 1] = shape.part(b).y
    return points


def get_126_landmarks_from_image(imgs, landmarks_network):
    r"""Get face keypoints from an image.

    Args:
        imgs (Nx3xHxW tensor or NxHxWx3 numpy array):
        Input images.
        landmarks_network (obj): The landmark detection network.

    Return:
        (Nx126x2 numpy array): Predicted landmarks.
    """
    if type(imgs) == torch.Tensor:
        imgs = ((imgs + 1) / 2 * 255).byte()
        imgs = np.transpose(imgs.cpu().numpy(), (0, 2, 3, 1))

    landmarks = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        out_boxes, landmark = \
            landmarks_network.get_face_boxes_and_landmarks(img)
        if len(landmark) > 1:
            # Pick the largest face in the image.
            face_size_max = face_index = 0
            for i, out_box in enumerate(out_boxes):
                face_size = max(out_box[2] - out_box[0],
                                out_box[1] - out_box[1])
                if face_size > face_size_max:
                    face_size_max = face_size
                    face_index = i
            landmark = landmark[face_index]
        elif len(landmark) == 1:
            landmark = landmark[0]
        else:
            landmark = np.zeros((126, 2), dtype=np.float32)
        landmarks += [landmark[np.newaxis]]
    landmarks = np.vstack(landmarks).astype(np.float32)
    return landmarks


def convert_face_landmarks_to_image(cfgdata, landmarks, output_size,
                                    output_tensor=True, cpu_only=False):
    r"""Convert the facial landmarks to a label map.

    Args:
        cfgdata (obj): Data configuration.
        landmarks
        output_size (tuple of int): H, W of output label map.
        output_tensor (bool): Output tensors instead of numpy arrays.
        cpu_only (bool): Output CPU tensor only.

    Returns:
        (NxCxHxW tensor or list of HxWxC numpy arrays): Label maps.
    """
    h, w = output_size
    labels = connect_face_keypoints(h, w, None, None, None, None, False,
                                    cfgdata, landmarks)
    if not output_tensor:
        return labels
    labels = [torch.from_numpy(label).permute(2, 0, 1).unsqueeze(0)
              for label in labels]
    labels = torch.cat(labels)
    if cpu_only:
        return labels
    return labels.cuda()


def add_face_keypoints(label_map, image, keypoints):
    r"""Add additional keypoints to label map.

    Args:
        label_map (Nx1xHxW tensor or None)
        image (Nx3xHxW tensor)
        keypoints (NxKx2 tensor)
    """
    if label_map is None:
        label_map = torch.zeros_like(image)[:, :1]
    x, y = keypoints[:, :, 0], keypoints[:, :, 1]
    h, w = image.shape[-2:]
    x = ((x + 1) / 2 * w).long()
    y = ((y + 1) / 2 * h).long()
    bs = torch.arange(label_map.shape[0]).cuda().view(-1, 1).expand_as(x)
    label_map[bs, :, y, x] = 1
    return label_map


def draw_edge(im, x, y, bw=1, color=(255, 255, 255), draw_end_points=False):
    r"""Set colors given a list of x and y coordinates for the edge.

    Args:
        im (HxWxC numpy array): Canvas to draw.
        x (1D numpy array): x coordinates of the edge.
        y (1D numpy array): y coordinates of the edge.
        bw (int): Width of the stroke.
        color (list or tuple of int): Color to draw.
        draw_end_points (bool): Whether to draw end points of the edge.
    """
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # Draw edge.
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h - 1, y + i))
                xx = np.maximum(0, np.minimum(w - 1, x + j))
                set_color(im, yy, xx, color)

        # Draw endpoints.
        if draw_end_points:
            for i in range(-bw * 2, bw * 2):
                for j in range(-bw * 2, bw * 2):
                    if (i ** 2) + (j ** 2) < (4 * bw ** 2):
                        yy = np.maximum(0, np.minimum(h - 1, np.array(
                            [y[0], y[-1]]) + i))
                        xx = np.maximum(0, np.minimum(w - 1, np.array(
                            [x[0], x[-1]]) + j))
                        set_color(im, yy, xx, color)


def set_color(im, yy, xx, color):
    r"""Set pixels of the image to the given color.

    Args:
        im (HxWxC numpy array): Canvas to draw.
        xx (1D numpy array): x coordinates of the pixels.
        yy (1D numpy array): y coordinates of the pixels.
        color (list or tuple of int): Color to draw.
    """
    if type(color) != list and type(color) != tuple:
        color = [color] * 3
    if len(im.shape) == 3 and im.shape[2] == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = \
                color[0], color[1], color[2]
        else:
            for c in range(3):
                im[yy, xx, c] = ((im[yy, xx, c].astype(float)
                                  + color[c]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]


def interp_points(x, y):
    r"""Given the start and end points, interpolate to get a curve/line.

    Args:
        x (1D array): x coordinates of the points to interpolate.
        y (1D array): y coordinates of the points to interpolate.

    Returns:
        (dict):
          - curve_x (1D array): x coordinates of the interpolated points.
          - curve_y (1D array): y coordinates of the interpolated points.
    """
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interp_points(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if len(x) < 3:
                    popt, _ = curve_fit(linear, x, y)
                else:
                    popt, _ = curve_fit(func, x, y)
                    if abs(popt[0]) > 1:
                        return None, None
            except Exception:
                return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], int(np.round(x[-1]-x[0])))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)


def func(x, a, b, c):
    r"""Quadratic fitting function."""
    return a * x**2 + b * x + c


def linear(x, a, b):
    r"""Linear fitting function."""
    return a * x + b
