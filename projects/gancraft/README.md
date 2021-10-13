# GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds

###
[Project](https://nvlabs.github.io/GANcraft/) |
[Paper](https://arxiv.org/abs/2104.07659) |
[Video](https://www.youtube.com/watch?v=1Hky092CGFQ) |
[Two Minute Papers Video](https://www.youtube.com/watch?v=jl0XCslxwB0)

<img src="https://nvlabs.github.io/GANcraft/videos/island_small.gif" alt="teaser" width="400"/>

## License

Imaginaire is released under [NVIDIA Software license](LICENSE.md).
For commercial use, please consult [researchinquiries@nvidia.com](researchinquiries@nvidia.com)


## Software Installation
For installation, please check out [INSTALL.md](../../INSTALL.md).

## Hardware Requirement
We trained our model using an NVIDIA DGX1 with 8 V100 32GB GPUs. Training took
about 4 days.


## Inference using pretrained model
Note: This code has been tested with the 21.06 docker (See `scripts/build_docker.sh` to build your own docker image). We do not guarantee correctness with other versions.

Download 3 sample voxel worlds `demoworld`, `landscape`, and `survivalisland` by running
```bash
python scripts/download_test_data.py --model_name gancraft
```

We have provided trained models for the 3 above worlds that will be automatically downloaded by running the following command, which performs inference.
```bash
python -m torch.distributed.launch --nproc_per_node=1 inference.py \
  --config configs/projects/gancraft/<world-name>.yaml \
  --output_dir projects/gancraft/output/<world-name>
```
The results are stored in `projects/gancraft/output/<world-name>`.

Various inference arguments can be specified or modified in the `inference_args` section of the config YAML file.

## Training

GANcraft requires a semantically-labeled voxel world as input. We have provided 3 sample worlds that can be downloaded with the command
```bash
python scripts/download_test_data.py --model_name gancraft
```

We also provide a tool for exporting your own Minecraft world with the help of [Mineways](https://www.realtimerendering.com/erich/minecraft/public/mineways/). To export your own world for use with GANcraft, select an area in Mineways, export it as schematic file (File -> Export Schematic), and convert the schematic file to Numpy npy file using the sch2vox tool provided. The tool is in `projects/gancraft/sch2vox`. You can then use the npy file in your custom GANcraft config file. For new worlds, we recommend specifying `preproc_ver: 6` in the config file.

It also needs a dataset of real images and segmentation maps as input, in order to apply a GAN loss between real images and network outputs. You can use the COCO dataset for this purpose. Please refer to [SPADE training data preparation](../spade/README.md#training-data-preparation) for instructions on how to prepare this.


### Training command

```bash
python -m torch.distributed.launch train.py \
  --config configs/projects/gancraft/demoworld.yaml
```
For pseudo-ground truth generation, we provide a SPADE model trained on the publicly available [LHQ dataset](https://github.com/universome/alis). This model will be automatically downloaded when the above command is run.
Note that in our paper, we trained our models using a SPADE model trained on a large-scale landscape dataset collected from the web, so results of your training might be different. We also sampled 24 points along each ray instead of 4 points as in the provided config file (which is configured to be compatible with GPUs with <12GB memory).

### Inference command
To perform inference using your own checkpoint, run
```bash
python -m torch.distributed.launch --nproc_per_node=1 inference.py \
  --config configs/projects/gancraft/demoworld.yaml \
  --output_dir projects/gancraft/output/demoworld \
  --checkpoint <path-to-your-checkpoint>
```

## Citation
If you use this code for your research, please cite

```
@inproceedings{hao2021GANcraft,
  title={{GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds}},
  author={Zekun Hao and Arun Mallya and Serge Belongie and Ming-Yu Liu},
  booktitle={ICCV},
  year={2021}
}
```
