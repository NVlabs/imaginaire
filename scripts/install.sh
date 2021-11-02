#!/bin/sh
CURRENT=$(pwd)

# Check CUDA_VERSION
export CUDA_VERSION=$(nvcc --version| grep -Po "(\d+\.)+\d+" | head -1)

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

pip install --upgrade -r scripts/requirements.txt

for p in correlation channelnorm resample2d bias_act upfirdn2d; do
  cd imaginaire/third_party/${p};
  rm -rf build dist *info;
  python setup.py install;
  cd ${CURRENT};
done

for p in gancraft/voxlib; do
  cd imaginaire/model_utils/${p};
  make all
  cd ${CURRENT};
done
