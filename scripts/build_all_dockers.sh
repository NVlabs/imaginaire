#!/bin/bash

y="21"

for m in 05 06 07
do
  rm -rf Dockerfile
  echo "FROM nvcr.io/nvidian/pytorch:${y}.${m}-py3" > Dockerfile
  input="Dockerfile.base"
  while IFS= read -r line
  do
    echo "$line" >> Dockerfile
  done < "$input"
  docker build -t imaginaire:${y}.${m}-py3 .
done
