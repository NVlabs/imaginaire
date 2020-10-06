<img src="imaginaire_logo.svg" alt="imaginaire_logo.svg" height="360"/>

# Imaginaire
### [Docs](http://imaginaire.cc/docs) | [License](LICENSE.md) | [Installation](INSTALL.md) | [Model Zoo](MODELZOO.md)

# Model Zoo

## Introduction

We provide a wide range of pretrained imaginaire models for different tasks. All the models were trained using an NVIDIA DGX 1 machine with 8 32GB V100 using NVIDIA PyTorch docker 20.03.


|Algorithm Name                               | Task                                                                                                            | Model        |  Resolution |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------|--------------|------------:|
|[pix2pixHD](projects/pix2pixHD/README.md)     | Cityscapes, segmentation to image                                                                               |[download](https://drive.google.com/file/d/1B3bXpQQzidJW0G3oCjYSWYEn2zd8h9dg/view?usp=sharing)  | 1024x512    |
|[SPADE](projects/spade/README.md)             | COCO-Stuff, segmentation to image                                                                               |[download](https://drive.google.com/file/d/1R27Zk9zlj8HitW_bOsQmbT2LOQNCKJJL/view?usp=sharing)  |  256x256    |


### Unsupervised Image-to-Image Translation


|Algorithm Name                               | Task                                                                                                            | Model        |  Resolution |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------|--------------|------------:|
|[UNIT](projects/unit/README.md)               | Winter <-> Summer                                                                                               |[download](https://drive.google.com/file/d/1y1FJT_kRq80Se6ASCU3LFZwrrJKzHr4I/view?usp=sharing)  | 256x256     |
|[MUNIT](projects/munit/README.md)             | AFHQ Dog <-> Cat                                                                                                |[download](https://drive.google.com/file/d/1XCqHFD1pN7Vlp0RWI0oYKpH4USSKdGqo/view?usp=sharing)  | 256x256     |
|[FUNIT](projects/funit/README.md)             | AniamlFaces                                                                                                     |[download](https://drive.google.com/file/d/1Tbq0zaaH_Omv_0IPfX8LIvh0sVmusqE-/view?usp=sharing)  | 256x256     |
|[COCO-FUNIT](projects/coco_funit/README.md)   | AniamlFaces                                                                                                     |[download](https://drive.google.com/file/d/1ODlwSfgauWyOSxj-aPCbFMOGUn2INiQT/view?usp=sharing)  | 256x256     |
|[COCO-FUNIT](projects/coco_funit/README.md)   | Mammals, Full body animal translation                                                                           |[download](https://drive.google.com/file/d/1Wf0BhcIpVJgHQunipdt8r-KtQ9mRvKxt/view?usp=sharing)  | 256x256     |


### Video-to-video Translation


|Algorithm Name                               | Task                                                                                                            | Model        |  Resolution |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------|--------------|------------:|
|[vid2vid](projects/vid2vid/README.md)         | Cityscapes, segmentation to video                                                                               |[download](https://drive.google.com/file/d/1b2M5rU740vBurLQ9iDP2kb4sP5HAb-Jx/view?usp=sharing)  | 1024x512    |
|[fs-vid2vid](projects/fs_vid2vid/README.md)   | FaceForensics, landmarks to video                                                                               |[download](https://drive.google.com/file/d/1F_22ctFmo553nRHy1d_BX7aorc9zk9cF/view?usp=sharing)  | 512x512     |
|[wc-vid2vid](projects/wc_vid2vid/README.md)   | Cityscapes, segmentation to video                                                                               |[download](https://drive.google.com/file/d/1CvRBok210WWQHF6VuZvU4Vuzdd05ItYB/view?usp=sharing)  | 1024x512    |
