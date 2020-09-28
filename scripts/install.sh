#!/bin/sh
CURRENT=$(pwd)

sudo apt-get update && sudo apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        git \
        curl \
        vim \
        tmux \
        wget \
        bzip2 \
        unzip \
        g++ \
        ca-certificates \
        ffmpeg \
        libx264-dev \
        imagemagick
pip install cmake --upgrade
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -y -c anaconda pyqt
pip install Pillow==6.1
pip install tensorboard
pip install scipy==1.3.3 --upgrade
pip install jupyterlab --upgrade
pip install scikit-image tqdm wget
pip install cython pyyaml lmdb
pip install opencv-python opencv-contrib-python
pip install open3d albumentations requests
pip install qimage2ndarray
pip install imageio-ffmpeg
pip install face-alignment dlib
pip install pynvml
pip install nvidia-ml-py3==7.352.0
pip install dlib
pip install imutils
pip install pyglet trimesh

cd imaginaire/third_party/correlation;
rm -rf build dist *info;
python setup.py install;
cd ${CURRENT};

cd imaginaire/third_party/channelnorm;
rm -rf build dist *info;
python setup.py install;
cd ${CURRENT};

cd imaginaire/third_party/resample2d;
rm -rf build dist *info;
python setup.py install
cd ${CURRENT};

cd /tmp;
rm -rf apex
git clone https://github.com/NVIDIA/apex;
cd apex;
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ${CURRENT};


pip install av
