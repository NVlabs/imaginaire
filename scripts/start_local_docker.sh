docker run \
    --gpus all \
    --shm-size 32g -it -u $(id -u):$(id -g) \
    -v /mnt:/mnt \
    -v ~/:/home \
    imaginaire:${1}-py3 \
    /bin/bash;
