# SPADE/GauGAN: Semantic Image Synthesis with Spatially-Adaptive Normalization

###
[Project](https://nvlabs.github.io/SPADE/) |
[Paper](https://arxiv.org/abs/1903.07291) |
[GTC Video (2m)](https://youtu.be/p5U4NgVGAwg) |
[Video (2m)](https://youtu.be/MXWm6w4E5q0) |
[Demo](https://www.nvidia.com/en-us/research/ai-playground/) |
[Previous Implementation](https://github.com/NVlabs/SPADE) |
[Two Minute Papers Video](https://youtu.be/hW1_Sidq3m8)

<img src="https://nvlabs.github.io/SPADE//images/ocean.gif" alt="teaser" width="400"/>

## License

Imaginaire is released under [NVIDIA Software license](LICENSE.md).
For commercial use, please consult [researchinquiries@nvidia.com](researchinquiries@nvidia.com)


## Software Installation
For installation, please checkout [INSTALL.md](../../INSTALL.md).

## Hardware Requirement
We trained our model using an NVIDIA DGX1 with 8 V100 32GB GPUs. Training took
about 2-3 week.

## Training

SPADE prefers the following data structure.
```
${TRAINING_DATASET_ROOT_FOLDER}
└───images
    └───0001.jpg
    └───0002.jpg
    └───0003.jpg
    ...
└───seg_maps
    └───0001.png
    └───0002.png
    └───0003.png
    ...
└───edge_maps
    └───0001.png
    └───0002.png
    └───0003.png
    ...
```

### Training data preparation

- Download
[COCO training images](http://images.cocodataset.org/zips/train2017.zip),
[COCO validation images](http://images.cocodataset.org/zips/val2017.zip)
, and [Stuff-Things map](http://calvin.inf.ed.ac
.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
the dataset and unzip the files.
Extract images, segmentation masks, and object boundaries for the edge maps.
Organize them based on the above data structure.

- Build the lmdbs
```bash
for f in train val; do
python scripts/build_lmdb.py \
--config configs/projects/spade/cocostuff/base128_bs4.yaml \
--data_root dataset/cocostuff_raw/${f} \
--output_root dataset/cocostuff/${f} \
--overwrite
done
```

### Training command

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--config configs/projects/spade/cocostuff/base128_bs4.yaml \
--logdir logs/projects/spade/cocostuff/base128_bs4.yaml
```

## Inference

SPADE prefers the following file arrangement for testing.
```
${TEST_DATASET_ROOT_FOLDER}
└───seg_maps
    └───0001.png
    └───0002.png
    └───0003.png
    ...
└───edge_maps
    └───0001.png
    └───0002.png
    └───0003.png
    ...
```

- Download sample test data by running
```bash
python scripts/download_test_data.py --model_name spade
```

```bash
python -m torch.distributed.launch --nproc_per_node=1 inference.py \
--config configs/projects/spade/cocostuff/base128_bs4.yaml \
--output_dir projects/spade/output/cocostuff
```

The results are stored in `projects/spade/output/cocostuff`

Below we show the expected output images.

<table>
  <tr>
    <td>
        Ground truth
    </td>
    <td>
        Segmentation
    </td>
    <td>
        Edge
    </td>
    <td>
        Synthesis result
    </td>
  </tr>
  <tr>
    <td>
    <img src="images/000000044195.jpg" alt="gt" width="256"/>
    </td>
    <td>
    <img src="seg_maps/000000044195.png" alt="seg" width="256"/>
    </td>
    <td>
    <img src="edge_maps/000000044195.png" alt="edge" width="256"/>
    </td>
    <td>
    <img src="results/000000044195.jpg" alt="result" width="256"/>
    </td>
  </tr>
  <tr>
    <td>
    <img src="images/000000058384.jpg" alt="gt" width="256"/>
    </td>
    <td>
    <img src="seg_maps/000000058384.png" alt="seg" width="256"/>
    </td>
    <td>
    <img src="edge_maps/000000058384.png" alt="edge" width="256"/>
    </td>
    <td>
    <img src="results/000000058384.jpg" alt="result" width="256"/>
    </td>
  </tr>
  <tr>
    <td>
    <img src="images/000000072795.jpg" alt="gt" width="256"/>
    </td>
    <td>
    <img src="seg_maps/000000072795.png" alt="seg" width="256"/>
    </td>
    <td>
    <img src="edge_maps/000000072795.png" alt="edge" width="256"/>
    </td>
    <td>
    <img src="results/000000072795.jpg" alt="result" width="256"/>
    </td>
  </tr>
</table>

## Citation
If you use this code for your research, please cite our papers.

```
@inproceedings{park2019SPADE,
  title={Semantic Image Synthesis with Spatially-Adaptive Normalization},
  author={Park, Taesung and Liu, Ming-Yu and Wang, Ting-Chun and Zhu, Jun-Yan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```