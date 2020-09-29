<img src="imaginaire_logo.svg" alt="imaginaire_logo.svg" height="360"/>

# Imaginaire
### [Docs](http://imaginaire.cc/docs) | [License](LICENSE.md) | [Installation](INSTALL.md) | [Model Zoo](MODELZOO.md)

# Installation Guide

Our library is developed using an Ubuntu 16.04 machine. We have not yet test our
  library on other operation systems.

## Prerequisite
- [Anaconda3](https://www.anaconda.com/products/individual)
- [cuda10.2](https://developer.nvidia.com/cuda-toolkit)
- [cudnn](https://developer.nvidia.com/cudnn) 

## Installation
```bash
git clone https://github.com/nvlabs/imaginaire
cd imaginaire
bash scripts/install.sh
bash scripts/test_training.sh
```
If installation is not successful, error message will be prompted.

## Docker Installation
We use NVIDIA docker image. We provide two ways to build the docker image.
1. Build a target docker image
```bash
bash scripts/build_docker.sh 20.05
```

2. Build many docker images in one command
```bash
bash scripts/build_all_dockers.sh
```
If installation is not successful, error message will be prompted.

3. Launch an interactive docker container and test the imaginaire repo.
```bash
cd docker
bash start_local_docker.sh 20.05
cd ${IMAGINAIRE_ROOT}
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
max-line-length = 80
```

## Windows Installation

- Install [git for windows](https://gitforwindows.org/)
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe)
- Install [CUDA10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
- Install [cudnn7.6.5](https://developer.nvidia.com/cudnn)
- Open a gitbash prompt
    - `cd YOUR_ROOT_DIR`
    - `git clone HTTP_PATH_TO_IMAGINAIRE`
    - `git clone https://github.com/NVIDIA/apex`
- Open an anaconda prompt.
    - `cd YOUR_ROOT_DIR\imaginaire2020`
    - `conda install -y pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch`
    - `conda install -y -c anaconda pyqt`
    - `scripts\install.bat`
    - `cd YOUR_ROOT_DIR\apex`
    - `pip install -v --no-cache-dir .`
- Run the code
    - Remember to set up the python path in the windows prompt before running any demo code from the repo.
        - `cd YOUR_ROOT_DIR\imaginaire2020`
        - `set PYTHONPATH=%PYTHONPATH%;%cd%`
