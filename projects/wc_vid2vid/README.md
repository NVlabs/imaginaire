# World Consistent vid2vid: Video-to-Video Synthesis
A GAN-based approach to generate 2D world renderings that are consistent over time and viewpoints. This method colors the 3D point cloud of the world as the camera moves through the world, coloring new regions in a manner consistent with the already colored world. It learns to render images based on the 2D projections of the point cloud to the camera in a semantically consistent manner while robustly dealing with incorrect and incomplete point clouds.

[Project](https://nvlabs.github.io/wc-vid2vid/) |
[YouTube](https://www.youtube.com/watch?v=b2P39sS2kKo) |
[arXiv](https://arxiv.org/abs/2007.08509) |
[Paper(full)](https://nvlabs.github.io/wc-vid2vid/files/wc-vid2vid.pdf) |
[Two Minute Papers Video](https://youtu.be/u4HpryLU-VI)

![](side_by_side.gif)

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

## Training
The following shows the example commands to train vid2vid on the Cityscapes dataset.
- Download the dataset and put it in the format as following.
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
└───unprojections
    └───seq0001
        └───000001.pkl
        └───000002.pkl
        ...
    └───seq0002
        └───000001.pkl
        └───000002.pkl
        ...
    ...
```
The scripts to perform SfM and generate the unprojection files are attached in this comment: https://github.com/NVlabs/imaginaire/issues/5#issuecomment-720146998.

- Preprocess the data into LMDB format

```bash
python scripts/build_lmdb.py --paired \
  --config configs/projects/wc_vid2vid/cityscapes/seg_ampO1.yaml \
  --data_root [PATH_TO_DATA train|val] \
  --output_root datasets/cityscapes/lmdb/[train|val]
```

- Train on 8 GPUs with AMPO1

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --config configs/projects/wc_vid2vid/cityscapes/seg_ampO1.yaml
```

## Inference
- Download some test data by running

```bash
python ./scripts/download_test_data.py --model_name wc_vid2vid
```

- Or arrange your own data into the same format as the training data described above.

- Translate segmentation masks to images
  - Inference command
    ```bash
    python inference.py --single_gpu \
      --config configs/projects/wc_vid2vid/cityscapes/seg_ampO1.yaml \
      --output_dir projects/wc_vid2vid/output/cityscapes
    ```
- The results are stored in `projects/wc_vid2vid/output/cityscapes`.
  Below, we show the expected output video.
<img alt="teaser" src='https://nvlabs.github.io/wc-vid2vid/videos/sample_output.gif' width='512'/>

## Citation
If you use this code for your research, please cite our papers.

```
@inproceedings{mallya2020world,
    title={World-Consistent Video-to-Video Synthesis},
    author={Arun Mallya and Ting-Chun Wang and Karan Sapra and Ming-Yu Liu},
    booktitle={European Conference on Computer Vision (ECCV)}},
    year={2020}
}
```
