# vid2vid: Video-to-Video Synthesis
Pytorch implementation for high-resolution (e.g., 2048x1024) photorealistic video-to-video translation. It can be used for turning semantic label maps into photo-realistic videos, synthesizing people talking from edge maps, or generating human motions from poses.

### 
[Project](https://tcwang0509.github.io/vid2vid/) |
[YouTube(short)](https://youtu.be/5zlcXTCpQqM) |
[YouTube(full)](https://youtu.be/GrP_aOSXt5U) |
[arXiv](https://arxiv.org/abs/1808.06601) |
[Paper(full)](https://tcwang0509.github.io/vid2vid/paper_vid2vid.pdf) |
[Previous Implementation](https://github.com/NVIDIA/vid2vid) |
[Two Minute Papers Video](https://youtu.be/GRQuRcpf5Gc)

<img src="https://github.com/NVIDIA/vid2vid/raw/master/imgs/teaser.gif" alt="teaser" width="640"/>

## License
Imaginaire is released under [NVIDIA Software license](LICENSE.md).
For commercial use, please consult [researchinquiries@nvidia.com](researchinquiries@nvidia.com)

## Software Installation
For installation, please checkout [INSTALL.md](../../INSTALL.md).

## Hardware Requirement
We trained our models using an NVIDIA DGX1 with 8 V100 32GB GPUs. You can try to use fewer GPUs or reduce the batch size if it does not fit in your GPU memory, but training stability and image quality cannot be guaranteed.

## Datasets

### Cityscapes
We use the Cityscapes dataset as an example. To train a model on the full dataset, please download it from the [official website](https://www.cityscapes-dataset.com/) (registration required). We apply a pre-trained segmentation algorithm to get the corresponding segmentation maps.

### Dancing
We use random dancing videos found on YouTube. You can also obtain a dancing dataset by simply recording a video of someone doing different motions for a few minutes. After that, please apply OpenPose on the frames to get the pose information.


## Training
The following shows the example commands to train vid2vid on the Cityscapes dataset. To train it on other datasets, replace all `cityscapes` with `dancing`.
- Download the dataset and put it in the format as following. For Cityscapes:
```
cityscapes
└───images
    └───seq0001
        └───000001.png
        └───000002.png
        ...
    └───seq0002
        └───000001.png
        └───000002.png
        ...
    ...
└───seg_maps
    └───seq0001
        └───000001.png
        └───000002.png
        ...
    └───seq0002
        └───000001.png
        └───000002.png
        ...
    ...
```

For the Dancing dataset:
```
dancing
└───images
    └───seq0001
        └───000001.jpg
        └───000002.jpg
        ...
└───poses-openpose
    └───seq0001
        └───000001.json
        └───000002.json
        ...
```

- Preprocess the data into LMDB format

```bash
python scripts/build_lmdb.py --config configs/projects/vid2vid/cityscapes/ampO1.yaml --data_root [PATH_TO_DATA] --output_root datasets/cityscapes/lmdb/[train | val] --paired
```

- Train on 8 GPUs with AMPO1

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py \
--config configs/projects/vid2vid/cityscapes/ampO1.yaml
```

## Inference
- Download some test data by running

```bash
python ./scripts/download_test_data.py --model_name vid2vid
```

- Or arrange your own data into the same format as the training data described above.

- Translate segmentation masks to images
  - Inference command
    ```bash
    python inference.py --single_gpu \
    --config configs/projects/vid2vid/cityscapes/ampO1.yaml \
    --output_dir projects/vid2vid/output/cityscapes
    ```

Below we show an example output video:

<img alt="output" src='output/cityscapes/stuttgart_00.gif' width='600'/>


## Citation
If you use this code for your research, please cite our papers.

```
@inproceedings{wang2018vid2vid,
   title     = {Video-to-Video Synthesis},
   author    = {Ting-Chun Wang and Ming-Yu Liu and Jun-Yan Zhu and Guilin Liu
                and Andrew Tao and Jan Kautz and Bryan Catanzaro},   
   booktitle = {Conference on Neural Information Processing Systems (NeurIPS)}},
   year      = {2018}
}
```
