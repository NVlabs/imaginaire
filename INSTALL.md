<img src="imaginaire_logo.svg" alt="imaginaire_logo.svg" height="360"/>

# Imaginaire
### [Docs](http://imaginaire.cc/docs) | [License](LICENSE.md) | [Installation](INSTALL.md) | [Model Zoo](MODELZOO.md)

# Installation Guide

Our library is developed using an Ubuntu 18.04 machine. We have not yet tested our library on other operating systems.

## Prerequisite
- [Anaconda3](https://www.anaconda.com/products/individual)
- [cuda11.1](https://developer.nvidia.com/cuda-toolkit)
- [cudnn](https://developer.nvidia.com/cudnn)

We provide three different ways of installing Imaginaire.

### 1. Installation (default)
Note: sudo privilege is required.
```bash
git clone https://github.com/nvlabs/imaginaire
cd imaginaire
bash scripts/install.sh
bash scripts/test_training.sh
```
If installation is not successful, error message will be prompted.

### 2. Installation with Docker
We use NVIDIA docker image. We provide two ways to build the docker image.
1. Build a target docker image
```bash
bash scripts/build_docker.sh 21.06
```

2. Launch an interactive docker container and test the imaginaire repo.
```bash
cd scripts
bash start_local_docker.sh 21.06
cd ${IMAGINAIRE_ROOT}
bash scripts/test_training.sh
```

### 3. Installation with Conda
Set up the Conda environment and install packages with
```bash
conda env create --file scripts/requirements_conda.yaml
# install third-party libraries
export CUDA_VERSION=$(nvcc --version| grep -Po "(\d+\.)+\d+" | head -1)
CURRENT=$(pwd)
for p in correlation channelnorm resample2d bias_act upfirdn2d; do
    cd imaginaire/third_party/${p};
    rm -rf build dist *info;
    python setup.py install;
    cd ${CURRENT};
done
```
To activate the environment and test the repo:
```bash
conda activate imaginaire
bash scripts/test_training.sh
```

## Flake8
We follow the PEP8 style using flake8. To follow our practice, please do
```bash
pip install flake8
flake8 --install-hook git
git config --bool flake8.strict true
```
We set the maximum line length to 80. To avoid error messages due to different line length, create a file `~/.config/flake8` with the following content:
```
[flake8]
max-line-length = 200
```

## Windows Installation [Out-dated]

- Install [git for windows](https://gitforwindows.org/)
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe)
- Install [CUDA11.1](https://developer.nvidia.com/cuda-11.1-download-archive)
- Install [cudnn](https://developer.nvidia.com/cudnn)
- Open an anaconda prompt.
```
cd https://github.com/NVlabs/imaginaire
.\scripts\install.bat
```

Powershell
```
$env:PYTHONPATH = pwd
Get-ChildItem Env:PYTHONPATH
```

