#!/bin/bash

key=${1}

rm -rf Dockerfile
echo "FROM nvcr.io/nvidia/pytorch:${key}-py3" > Dockerfile
input="Dockerfile.base"

while IFS= read -r line
do
  echo "$line" >> Dockerfile
done < "$input"

input="scripts/requirements.txt"
while IFS= read -r line
do
  echo "RUN pip install $line" >> Dockerfile
done < "$input"


for p in correlation channelnorm resample2d bias_act upfirdn2d; do
  echo "COPY imaginaire/third_party/$p $p" >> Dockerfile
  echo "RUN cd $p && rm -rf build dist *-info && python setup.py install" >> Dockerfile
done

# Compile GANcraft libraries.
echo "COPY imaginaire/model_utils/gancraft/voxlib gancraft/voxlib" >> Dockerfile
echo "RUN cd gancraft/voxlib && make" >> Dockerfile

docker build -t nvcr.io/nvidian/lpr-imagine/imaginaire:${key}-py3 .
