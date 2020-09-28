# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from torch import nn

from imaginaire.discriminators.multires_patch import \
    WeightSharedMultiResPatchDiscriminator
from imaginaire.discriminators.residual import ResDiscriminator


class Discriminator(nn.Module):
    r"""UNIT discriminator. It can be either a multi-resolution patch
    discriminator like in the original implementation, or a
    global residual discriminator.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, dis_cfg, data_cfg):
        super().__init__()
        if getattr(dis_cfg, 'patch_dis', True):
            # Use the multi-resolution patch discriminator. It works better for
            # scene images and when you want to preserve pixel-wise
            # correspondence during translation.
            self.discriminator_a = \
                WeightSharedMultiResPatchDiscriminator(**vars(dis_cfg))
            self.discriminator_b = \
                WeightSharedMultiResPatchDiscriminator(**vars(dis_cfg))
        else:
            # Use the global residual discriminator. It works better if images
            # have a single centered object (e.g., animal faces, shoes).
            self.discriminator_a = ResDiscriminator(**vars(dis_cfg))
            self.discriminator_b = ResDiscriminator(**vars(dis_cfg))

    def forward(self, data, net_G_output, gan_recon=False, real=True):
        r"""Returns the output of the discriminator.

        Args:
            data (dict):
              - images_a  (tensor) : Images in domain A.
              - images_b  (tensor) : Images in domain B.
            net_G_output (dict):
              - images_ab  (tensor) : Images translated from domain A to B by
                the generator.
              - images_ba  (tensor) : Images translated from domain B to A by
                the generator.
              - images_aa  (tensor) : Reconstructed images in domain A.
              - images_bb  (tensor) : Reconstructed images in domain B.
            gan_recon (bool): If ``True``, also classifies reconstructed images.
            real (bool): If ``True``, also classifies real images. Otherwise it
                only classifies generated images to save computation during the
                generator update.
        Returns:
            (dict):
              - out_ab (tensor): Output of the discriminator for images
                translated from domain A to B by the generator.
              - out_ab (tensor): Output of the discriminator for images
                translated from domain B to A by the generator.
              - fea_ab (tensor): Intermediate features of the discriminator
                for images translated from domain B to A by the generator.
              - fea_ba (tensor): Intermediate features of the discriminator
                for images translated from domain A to B by the generator.

              - out_a (tensor): Output of the discriminator for images
                in domain A.
              - out_b (tensor): Output of the discriminator for images
                in domain B.
              - fea_a (tensor): Intermediate features of the discriminator
                for images in domain A.
              - fea_b (tensor): Intermediate features of the discriminator
                for images in domain B.

              - out_aa (tensor): Output of the discriminator for
                reconstructed images in domain A.
              - out_bb (tensor): Output of the discriminator for
                reconstructed images in domain B.
              - fea_aa (tensor): Intermediate features of the discriminator
                for reconstructed images in domain A.
              - fea_bb (tensor): Intermediate features of the discriminator
                for reconstructed images in domain B.
        """
        out_ab, fea_ab, _ = self.discriminator_b(net_G_output['images_ab'])
        out_ba, fea_ba, _ = self.discriminator_a(net_G_output['images_ba'])
        output = dict(out_ba=out_ba, out_ab=out_ab,
                      fea_ba=fea_ba, fea_ab=fea_ab)
        if real:
            out_a, fea_a, _ = self.discriminator_a(data['images_a'])
            out_b, fea_b, _ = self.discriminator_b(data['images_b'])
            output.update(dict(out_a=out_a, out_b=out_b,
                               fea_a=fea_a, fea_b=fea_b))
        if gan_recon:
            out_aa, fea_aa, _ = self.discriminator_a(net_G_output['images_aa'])
            out_bb, fea_bb, _ = self.discriminator_b(net_G_output['images_bb'])
            output.update(dict(out_aa=out_aa, out_bb=out_bb,
                               fea_aa=fea_aa, fea_bb=fea_bb))
        return output
