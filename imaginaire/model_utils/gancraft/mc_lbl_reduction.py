# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os
import csv


class ReducedLabelMapper:
    def __init__(self):
        this_path = os.path.dirname(os.path.abspath(__file__))
        print('[ReducedLabelMapper] Loading from {}'.format(this_path))

        # Load Minecraft LUT
        mcid2rdlbl_lut = {}
        mcid2mclbl_lut = {}
        with open(os.path.join(this_path, 'mc_reduction.csv'), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                mcid = int(row[0])
                mcid2rdlbl_lut[mcid] = row[3]
                mcid2mclbl_lut[mcid] = row[1]

        # Load reduced label set
        reduced_lbls = []
        rdlbl2rdid = {}
        with open(os.path.join(this_path, 'reduced_coco_lbls.csv'), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(csvreader):
                rdlbl2rdid[row[0]] = idx
                reduced_lbls.append(row[0])
        print(['{}: {}'.format(rdid, rdlbl) for rdid, rdlbl in enumerate(reduced_lbls)])
        # The first label should always be 'ignore'
        assert reduced_lbls[0] == 'ignore'

        # Generate Minecraft ID to Reduced ID LUT
        mcid2rdid_lut = []
        for mcid in range(len(mcid2rdlbl_lut)):
            rdlbl = mcid2rdlbl_lut[mcid]
            if rdlbl == '':
                rdlbl = 'ignore'
            rdid = rdlbl2rdid[rdlbl]
            mcid2rdid_lut.append(rdid)

        # ================= coco part ==================
        gg_label_list = []
        gglbl2ggid = {}
        with open(os.path.join(this_path, 'gaugan_lbl2col.csv'), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(csvreader):
                gg_label_list.append(row[0])
                gglbl2ggid[row[0]] = idx

        # Load coco -> reduced mapping table
        gglbl2rdid = {}
        with open(os.path.join(this_path, 'gaugan_reduction.csv'), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(csvreader):
                gglbl = row[0]
                target_rdlbl = row[1]
                ggid = gglbl2ggid[gglbl]
                target_rdid = rdlbl2rdid[target_rdlbl]
                gglbl2rdid[ggid] = target_rdid
        ggid2rdid = [gglbl2rdid[i] for i in range(len(gglbl2rdid))]

        print('[ReducedLabelMapper] #Reduced Labels: {}'.format(len(reduced_lbls)))

        self.mcid2rdid_lut = mcid2rdid_lut
        self.ggid2rdid = ggid2rdid
        self.reduced_lbls = reduced_lbls

        self.ignore_id = rdlbl2rdid['ignore']
        self.dirt_id = rdlbl2rdid['dirt']
        self.water_id = rdlbl2rdid['water']

        self.gglbl2ggid = gglbl2ggid

    def gglbl2ggid(self, gglbl):
        return self.gglbl2ggid[gglbl]


if __name__ == '__main__':
    mapper = ReducedLabelMapper()
