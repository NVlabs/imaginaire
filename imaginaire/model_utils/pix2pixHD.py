# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""Utils for the pix2pixHD model."""
import numpy as np
import torch

from imaginaire.utils.data import get_paired_input_label_channel_number
from imaginaire.utils.distributed import dist_all_gather_tensor, is_master
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.trainer import (get_optimizer, get_optimizer_for_params,
                                      wrap_model_and_optimizer)
from sklearn.cluster import KMeans


def cluster_features(cfg, train_data_loader, net_E,
                     preprocess=None, small_ratio=0.0625, is_cityscapes=True):
    r"""Use clustering to compute the features.

    Args:
        cfg (obj): Global configuration file.
        train_data_loader (obj): Dataloader for iterate through the training
            set.
        net_E (nn.Module): Pytorch network.
        preprocess (function): Pre-processing function.
        small_ratio (float): We only consider instance that at least occupy
            $(small_ratio) amount of image space.
        is_cityscapes (bool): Is this is the cityscape dataset? In the
            Cityscapes dataset, the instance labels for car start with 26001,
            26002, ...

    Returns:
        ( num_labels x num_cluster_centers x feature_dims): cluster centers.
    """
    # Encode features.
    label_nc = get_paired_input_label_channel_number(cfg.data)
    feat_nc = cfg.gen.enc.num_feat_channels
    n_clusters = getattr(cfg.gen.enc, 'num_clusters', 10)
    # Compute features.
    features = {}
    for label in range(label_nc):
        features[label] = np.zeros((0, feat_nc + 1))
    for data in train_data_loader:
        if preprocess is not None:
            data = preprocess(data)
        feat = encode_features(net_E, feat_nc, label_nc,
                               data['images'], data['instance_maps'],
                               is_cityscapes)
        # We only collect the feature vectors for the master GPU.
        if is_master():
            for label in range(label_nc):
                features[label] = np.append(
                    features[label], feat[label], axis=0)
    # Clustering.
    # We only perform clustering for the master GPU.
    if is_master():
        for label in range(label_nc):
            feat = features[label]
            # We only consider segments that are greater than a pre-set
            # threshold.
            feat = feat[feat[:, -1] > small_ratio, :-1]
            if feat.shape[0]:
                n_clusters = min(feat.shape[0], n_clusters)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feat)
                n, d = kmeans.cluster_centers_.shape
                this_cluster = getattr(net_E, 'cluster_%d' % label)
                this_cluster[0:n, :] = torch.Tensor(
                    kmeans.cluster_centers_).float()


def encode_features(net_E, feat_nc, label_nc, image, inst,
                    is_cityscapes=True):
    r"""Compute feature embeddings for an image image.
    TODO(Ting-Chun): To make this funciton dataset independent.

    Args:
        net_E (nn.Module): The encoder network.
        feat_nc (int): Feature dimensions
        label_nc (int): Number of segmentation labels.
        image (tensor): Input image tensor.
        inst (tensor): Input instance map.
        is_cityscapes (bool): Is this is the cityscape dataset? In the
            Cityscapes dataset, the instance labels for car start with 26001,
            26002, ...
    Returns:
        (list of list of numpy vectors): We will have $(label_nc)
            list. For each list, it will record a list of feature vectors of
            dimension $(feat_nc+1) where the first $(feat_nc) dimensions is
            the representative feature of an instance and the last dimension
            is the proportion.
    """
    # h, w = inst.size()[2:]
    feat_map = net_E(image, inst)
    feature_map_gather = dist_all_gather_tensor(feat_map)
    inst_gathered = dist_all_gather_tensor(inst)
    # Initialize the cluster centers.
    # For each feature vector,
    #   0:feat_nc will be the feature vector.
    #   The feat_nc dimension record the percentage of the instance.
    feature = {}
    for i in range(label_nc):
        feature[i] = np.zeros((0, feat_nc + 1))
    if is_master():
        all_feat_map = torch.cat(feature_map_gather, 0)
        all_inst_map = torch.cat(inst_gathered, 0)
        # Scan through the batches.
        for n in range(all_feat_map.size()[0]):
            feat_map = all_feat_map[n:(n + 1), :, :, :]
            inst = all_inst_map[n:(n + 1), :, :, :]
            fh, fw = feat_map.size()[2:]
            inst_np = inst.cpu().numpy().astype(int)
            for i in np.unique(inst_np):
                if is_cityscapes:
                    label = i if i < 1000 else i // 1000
                else:
                    label = i
                idx = (inst == int(i)).nonzero()
                num = idx.size()[0]
                # We will just pick the middle pixel as its representative
                # feature.
                idx = idx[num // 2, :]
                val = np.zeros((1, feat_nc + 1))
                for k in range(feat_nc):
                    # We expect idx[0]=0 and idx[1]=0 as the number of sample
                    # per processing is 1 (idx[0]=0) and the channel number of
                    # the instance map is 1.
                    val[0, k] = feat_map[
                        idx[0], idx[1] + k, idx[2], idx[3]].item()
                val[0, feat_nc] = float(num) / (fh * fw)
                feature[label] = np.append(feature[label], val, axis=0)
        return feature
    else:
        return feature


def get_edges(t):
    r""" Compute edge maps for a given input instance map.

    Args:
        t (4D tensor): Input instance map.
    Returns:
        (4D tensor): Output edge map.
    """
    edge = torch.cuda.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (
        t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (
        t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (
        t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (
        t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
    return edge.float()


def get_optimizer_with_params(cfg, net_G, net_D, param_names_start_with=[],
                              param_names_include=[]):
    r"""Return the optimizer object.

    Args:
        cfg (obj): Global config.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        param_names_start_with (list of strings): Params whose names
            start with any of the strings will be trained.
        param_names_include (list of strings): Params whose names include
            any of the strings will be trained.
    """
    def get_train_params(net, param_names_start_with, param_names_include):
        r"""Get train parameters.

        Args:
            net (obj): Network object.
            param_names_start_with (list of strings): Params whose names
                start with any of the strings will be trained.
            param_names_include (list of strings): Params whose names include
                any of the strings will be trained.
        """
        params_to_train = []
        params_dict = net.state_dict()
        list_of_param_names_to_train = set()
        # Iterate through all params in the network and check if we need to
        # train it.
        for key, value in params_dict.items():
            do_train = False
            # If the param name starts with the target string (excluding
            # the 'module' part etc), we will train this param.
            key_s = key.replace('module.', '').replace('averaged_model.', '')
            for param_name in param_names_start_with:
                if key_s.startswith(param_name):
                    do_train = True
                    list_of_param_names_to_train.add(param_name)

            # Otherwise, if the param name includes the target string,
            # we will also train it.
            if not do_train:
                for param_name in param_names_include:
                    if param_name in key_s:
                        do_train = True
                        full_param_name = \
                            key_s[:(key_s.find(param_name) + len(param_name))]
                        list_of_param_names_to_train.add(full_param_name)

            # If we decide to train the param, add it to the list to train.
            if do_train:
                module = net
                key_list = key.split('.')
                for k in key_list:
                    module = getattr(module, k)
                params_to_train += [module]

        print('Training layers: ', sorted(list_of_param_names_to_train))
        return params_to_train

    # If any of the param name lists is not empty, will only train
    # these params. Otherwise will train the entire network (all params).
    if param_names_start_with or param_names_include:
        params = get_train_params(net_G, param_names_start_with,
                                  param_names_include)
    else:
        params = net_G.parameters()

    opt_G = get_optimizer_for_params(cfg.gen_opt, params)
    opt_D = get_optimizer(cfg.dis_opt, net_D)
    return wrap_model_and_optimizer(cfg, net_G, net_D, opt_G, opt_D)
