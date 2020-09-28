# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
import numpy as np
import random
import importlib
from .common import tensor2im, tensor2label
from .face import draw_edge, interp_points
from imaginaire.model_utils.fs_vid2vid import extract_valid_pose_labels


def draw_openpose_npy(resize_h, resize_w, crop_h, crop_w, original_h,
                      original_w, is_flipped, cfgdata, keypoints_npy):
    r"""Connect the OpenPose keypoints to edges and draw the pose map.

    Args:
        resize_h (int): Height the input image was resized to.
        resize_w (int): Width the input image was resized to.
        crop_h (int): Height the input image was cropped.
        crop_w (int): Width the input image was cropped.
        original_h (int): Original height of the input image.
        original_w (int): Original width of the input image.
        is_flipped (bool): Is the input image flipped.
        cfgdata (obj): Data configuration.
        keypoints_npy (dict): OpenPose keypoint dict.

    Returns:
        (list of HxWxC numpy array): Drawn label map.
    """
    pose_cfg = cfgdata.for_pose_dataset
    # Whether to draw only the basic keypoints.
    basic_points_only = getattr(pose_cfg, 'basic_points_only', False)
    # Whether to remove the face labels to avoid overfitting.
    remove_face_labels = getattr(pose_cfg, 'remove_face_labels', False)
    # Whether to randomly drop some keypoints to avoid overfitting.
    random_drop_prob = getattr(pose_cfg, 'random_drop_prob', 0)

    # Get the list of edges to draw.
    edge_lists = define_edge_lists(basic_points_only)
    op_key = cfgdata.keypoint_data_types[0]
    for input_type in cfgdata.input_types:
        if op_key in input_type:
            nc = input_type[op_key].num_channels
    if crop_h is not None:
        h, w = crop_h, crop_w
    else:
        h, w = resize_h, resize_w

    outputs = []
    for keypoint_npy in keypoints_npy:
        person_keypoints = np.asarray(keypoint_npy).reshape(-1, 137, 3)[0]
        # Separate out the keypoint array to different parts.
        pose_pts = person_keypoints[:25]
        face_pts = person_keypoints[25: (25 + 70)]
        hand_pts_l = person_keypoints[(25 + 70): (25 + 70 + 21)]
        hand_pts_r = person_keypoints[-21:]
        all_pts = [pose_pts, face_pts, hand_pts_l, hand_pts_r]
        # Remove the keypoints with low confidence.
        all_pts = [extract_valid_keypoints(pts, edge_lists)
                   for pts in all_pts]

        # Connect the keypoints to form the label map.
        pose_img = connect_pose_keypoints(all_pts, edge_lists,
                                          (h, w, nc),
                                          basic_points_only,
                                          remove_face_labels,
                                          random_drop_prob)
        pose_img = pose_img.astype(np.float32) / 255.0
        outputs.append(pose_img)
    return outputs


def openpose_to_npy_largest_only(inputs):
    r"""Convert OpenPose dicts to numpy arrays of keypoints. Only return the
    largest/tallest person in each dict.

    Args:
        inputs (list of dicts): List of OpenPose dicts.

    Returns:
        (list of numpy arrays): Keypoints.
    """
    return base_openpose_to_npy(inputs, return_largest_only=True)


def openpose_to_npy(inputs):
    r"""Conver OpenPose dicts to numpy arrays of keypoints.

    Args:
        inputs (list of dicts): List of OpenPose dicts.

    Returns:
        (list of numpy arrays): Keypoints.
    """
    return base_openpose_to_npy(inputs, return_largest_only=False)


def base_openpose_to_npy(inputs, return_largest_only=False):
    r"""Convert OpenPose dicts to numpy arrays of keypoints.

    Args:
        inputs (list of dicts): List of OpenPose dicts.
        return_largest_only (bool): Whether to return only the largest person.

    Returns:
        (list of numpy arrays): Keypoints.
    """
    outputs_npy = []
    for input in inputs:
        people_dict = input['people']
        n_ppl = max(1, len(people_dict))
        output_npy = np.zeros((n_ppl, 25 + 70 + 21 + 21, 3), dtype=np.float32)
        y_len_max = 0
        for i, person_dict in enumerate(people_dict):
            # Extract corresponding keypoints from the dict.
            pose_pts = np.array(person_dict["pose_keypoints_2d"]).reshape(25, 3)
            face_pts = np.array(person_dict["face_keypoints_2d"]).reshape(70, 3)
            hand_pts_l = np.array(person_dict["hand_left_keypoints_2d"]
                                  ).reshape(21, 3)
            hand_pts_r = np.array(person_dict["hand_right_keypoints_2d"]
                                  ).reshape(21, 3)

            if return_largest_only:
                # Get the body length.
                y = pose_pts[pose_pts[:, 2] > 0.01, 1]
                y_len = y.max() - y.min()
                if y_len > y_len_max:
                    y_len_max = y_len
                    max_ind = i

            # Concatenate all keypoint together.
            output_npy[i] = np.vstack([pose_pts, face_pts,
                                       hand_pts_l, hand_pts_r])
        if return_largest_only:
            # Only return the largest person in the dict.
            output_npy = output_npy[max_ind: max_ind + 1]

        outputs_npy += [output_npy.astype(np.float32)]
    return outputs_npy


def extract_valid_keypoints(pts, edge_lists):
    r"""Use only the valid keypoints by looking at the detection confidences.
    If the confidences for all keypoints in an edge are above threshold,
    keep the keypoints. Otherwise, their coordinates will be set to zero.

    Args:
        pts (Px3 numpy array): Keypoint xy coordinates + confidence.
        edge_lists (nested list of ints):  List of keypoint indices for edges.

    Returns:
        (Px2 numpy array): Output keypoints.
    """
    pose_edge_list, _, hand_edge_list, _, face_list = edge_lists
    p = pts.shape[0]
    thre = 0.1 if p == 70 else 0.01
    output = np.zeros((p, 2))

    if p == 70:  # ai_emoji
        for edge_list in face_list:
            for edge in edge_list:
                if (pts[edge, 2] > thre).all():
                    output[edge, :] = pts[edge, :2]
    elif p == 21:  # hand
        for edge in hand_edge_list:
            if (pts[edge, 2] > thre).all():
                output[edge, :] = pts[edge, :2]
    else:  # pose
        valid = (pts[:, 2] > thre)
        output[valid, :] = pts[valid, :2]

    return output


def connect_pose_keypoints(pts, edge_lists, size, basic_points_only,
                           remove_face_labels, random_drop_prob):
    r"""Draw edges by connecting the keypoints onto the label map.

    Args:
        pts (Px3 numpy array): Keypoint xy coordinates + confidence.
        edge_lists (nested list of ints):  List of keypoint indices for edges.
        size (tuple of int): Output size.
        basic_points_only (bool): Whether to use only the basic keypoints.
        remove_face_labels (bool): Whether to remove face labels.
        random_drop_prob (float): Probability to randomly drop keypoints.

    Returns:
        (HxWxC numpy array): Output label map.
    """
    pose_pts, face_pts, hand_pts_l, hand_pts_r = pts
    h, w, c = size
    body_edges = np.zeros((h, w, c), np.uint8)
    # If using one-hot, different parts of the body will be drawn to
    # different channels.
    use_one_hot = c > 3
    if use_one_hot:
        assert c == 27
    pose_edge_list, pose_color_list, hand_edge_list, hand_color_list, \
        face_list = edge_lists

    # Draw pose edges.
    h = int(pose_pts[:, 1].max() - pose_pts[:, 1].min())
    bw = max(1, h // 150)  # Stroke width.
    body_edges = draw_edges(body_edges, pose_pts, [pose_edge_list], bw,
                            use_one_hot, random_drop_prob,
                            colors=pose_color_list, draw_end_points=True)

    if not basic_points_only:
        # Draw hand edges.
        bw = max(1, h // 450)
        for i, hand_pts in enumerate([hand_pts_l, hand_pts_r]):
            if use_one_hot:
                k = 24 + i
                body_edges[:, :, k] = draw_edges(body_edges[:, :, k], hand_pts,
                                                 [hand_edge_list],
                                                 bw, False, random_drop_prob,
                                                 colors=[255] * len(hand_pts))
            else:
                body_edges = draw_edges(body_edges, hand_pts, [hand_edge_list],
                                        bw, False, random_drop_prob,
                                        colors=hand_color_list)
        # Draw face edges.
        if not remove_face_labels:
            if use_one_hot:
                k = 26
                body_edges[:, :, k] = draw_edges(body_edges[:, :, k], face_pts,
                                                 face_list, bw, False,
                                                 random_drop_prob)
            else:
                body_edges = draw_edges(body_edges, face_pts, face_list, bw,
                                        False, random_drop_prob)
    return body_edges


def draw_edges(canvas, keypoints, edges_list, bw, use_one_hot,
               random_drop_prob=0, edge_len=2, colors=None,
               draw_end_points=False):
    r"""Draw all the edges in the edge list on the canvas.

    Args:
        canvas (HxWxK numpy array): Canvas to draw.
        keypoints (Px2 numpy array): Keypoints.
        edge_list (nested list of ints):  List of keypoint indices for edges.
        bw (int): Stroke width.
        use_one_hot (bool): Use one-hot encoding or not.
        random_drop_prob (float): Probability to randomly drop keypoints.
        edge_len (int): Number of keypoints in an edge.
        colors (tuple of int): Color to draw.
        draw_end_points (bool): Whether to draw end points for edges.

    Returns:
        (HxWxK numpy array): Output.
    """
    k = 0
    for edge_list in edges_list:
        for i, edge in enumerate(edge_list):
            for j in range(0, max(1, len(edge) - 1), edge_len - 1):
                if random.random() > random_drop_prob:
                    sub_edge = edge[j:j + edge_len]
                    x, y = keypoints[sub_edge, 0], keypoints[sub_edge, 1]
                    if 0 not in x:  # Get rid of invalid keypoints.
                        curve_x, curve_y = interp_points(x, y)
                        if use_one_hot:
                            # If using one-hot, draw to different channels of
                            # the canvas.
                            draw_edge(canvas[:, :, k], curve_x, curve_y,
                                      bw=bw, color=255,
                                      draw_end_points=draw_end_points)
                        else:
                            color = colors[i] if colors is not None \
                                else (255, 255, 255)
                            draw_edge(canvas, curve_x, curve_y,
                                      bw=bw, color=color,
                                      draw_end_points=draw_end_points)
                k += 1
    return canvas


def define_edge_lists(basic_points_only):
    r"""Define the list of keypoints that should be connected to form the edges.

    Args:
        basic_points_only (bool): Whether to use only the basic keypoints.
    """
    # Pose edges and corresponding colors.
    pose_edge_list = [
        [17, 15], [15, 0], [0, 16], [16, 18],  # head
        [0, 1], [1, 8],                        # body
        [1, 2], [2, 3], [3, 4],                # right arm
        [1, 5], [5, 6], [6, 7],                # left arm
        [8, 9], [9, 10], [10, 11],             # right leg
        [8, 12], [12, 13], [13, 14]            # left leg
    ]
    pose_color_list = [
        [153, 0, 153], [153, 0, 102], [102, 0, 153], [51, 0, 153],
        [153, 0, 51], [153, 0, 0],
        [153, 51, 0], [153, 102, 0], [153, 153, 0],
        [102, 153, 0], [51, 153, 0], [0, 153, 0],
        [0, 153, 51], [0, 153, 102], [0, 153, 153],
        [0, 102, 153], [0, 51, 153], [0, 0, 153],
    ]

    if not basic_points_only:
        pose_edge_list += [
            [11, 24], [11, 22], [22, 23],  # right foot
            [14, 21], [14, 19], [19, 20]   # left foot
        ]
        pose_color_list += [
            [0, 153, 153], [0, 153, 153], [0, 153, 153],
            [0, 0, 153], [0, 0, 153], [0, 0, 153]
        ]

    # Hand edges and corresponding colors.
    hand_edge_list = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    hand_color_list = [
        [204, 0, 0], [163, 204, 0], [0, 204, 82], [0, 82, 204], [163, 0, 204]
    ]

    # Face edges.
    face_list = [
        [range(0, 17)],   # face contour
        [range(17, 22)],  # left eyebrow
        [range(22, 27)],  # right eyebrow
        [[28, 31], range(31, 36), [35, 28]],   # nose
        [[36, 37, 38, 39], [39, 40, 41, 36]],  # left eye
        [[42, 43, 44, 45], [45, 46, 47, 42]],  # right eye
        [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
    ]

    return pose_edge_list, pose_color_list, hand_edge_list, hand_color_list, \
        face_list


def tensor2pose(cfg, label_tensor):
    r"""Convert output tensor to a numpy pose map.

    Args:
        label_tensor (3D/4D/5D tensor): Label tensor.

    Returns:
        (HxWx3 numpy array or list of numpy arrays): Pose map.
    """
    if label_tensor.dim() == 5 or label_tensor.dim() == 4:
        return [tensor2pose(cfg, label_tensor[idx])
                for idx in range(label_tensor.size(0))]

    # If adding additional discriminators, draw the bbox for the regions
    # (e.g. faces) too.
    add_dis_cfg = getattr(cfg.dis, 'additional_discriminators', None)
    if add_dis_cfg is not None:
        crop_coords = []
        for name in add_dis_cfg:
            v = add_dis_cfg[name].vis
            file, crop_func = v.split('::')
            file = importlib.import_module(file)
            crop_func = getattr(file, crop_func)
            crop_coord = crop_func(cfg.data, label_tensor)
            if len(crop_coord) > 0:
                if type(crop_coord[0]) == list:
                    crop_coords.extend(crop_coord)
                else:
                    crop_coords.append(crop_coord)

    pose_cfg = cfg.data.for_pose_dataset
    pose_type = getattr(pose_cfg, 'pose_type', 'both')
    remove_face_labels = getattr(pose_cfg, 'remove_face_labels', False)
    label_tensor = extract_valid_pose_labels(label_tensor, pose_type,
                                             remove_face_labels)

    # If using both DensePose and OpenPose, overlay one image onto the other
    # to get the visualization map.
    dp_key = 'pose_maps-densepose'
    op_key = 'poses-openpose'
    use_densepose = use_openpose = False
    for input_type in cfg.data.input_types:
        if dp_key in input_type:
            dp_ch = input_type[dp_key].num_channels
            use_densepose = True
        elif op_key in input_type:
            op_ch = input_type[op_key].num_channels
            use_openpose = True
    if use_densepose:
        label_img = tensor2im(label_tensor[:dp_ch])
    if use_openpose:
        openpose = label_tensor[-op_ch:]
        openpose = tensor2im(openpose) if op_ch == 3 else \
            tensor2label(openpose, op_ch)
        if use_densepose:
            label_img[openpose != 0] = openpose[openpose != 0]
        else:
            label_img = openpose

    # Draw the bbox for the regions for the additional discriminator.
    if add_dis_cfg is not None:
        for crop_coord in crop_coords:
            ys, ye, xs, xe = crop_coord
            label_img[ys, xs:xe, :] = label_img[ye - 1, xs:xe, :] \
                = label_img[ys:ye, xs, :] = label_img[ys:ye, xe - 1, :] = 255

    if len(label_img.shape) == 2:
        label_img = np.repeat(label_img[:, :, np.newaxis], 3, axis=2)
    return label_img
