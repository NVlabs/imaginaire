docker run \
    --gpus all \
    --shm-size 32g \
    --ipc=host \
    -it \
    -v /mnt:/mnt \
    -v ~/:/home \
    nvcr.io/nvidian/lpr-imagine/imaginaire:${1}-py3 \
    /bin/bash

