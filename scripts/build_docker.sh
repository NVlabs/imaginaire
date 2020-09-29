#!/bin/bash

key=${1}

rm -rf Dockerfile
echo "FROM nvcr.io/nvidia/pytorch:${key}-py3" > Dockerfile
input="Dockerfile.base"
while IFS= read -r line
do
  echo "$line" >> Dockerfile
done < "$input"
docker build -t imaginaire:${key}-py3 .
