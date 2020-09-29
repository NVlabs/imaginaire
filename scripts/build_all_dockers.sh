#!/bin/bash

y="20"

for m in 05 06 07 08
do
  rm -rf Dockerfile
  echo "FROM nvcr.io/nvidia/pytorch:${y}.${m}-py3" > Dockerfile
  input="Dockerfile.base"
  while IFS= read -r line
  do
    echo "$line" >> Dockerfile
  done < "$input"
  docker build -t imaginaire:${y}.${m}-py3 .
done
