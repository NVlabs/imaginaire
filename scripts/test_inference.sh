#!/bin/bash
function output {
  eval ${cmd}
  RESULT=$?
  if [ $RESULT -eq 0 ]; then
    echo -e "\e[1;32m ${cmd} [Success] \e[0m"
  else
    echo -e "\e[1;31m ${cmd} [Failure] \e[0m"
    exit 1
  fi
}


rm projects/*/*.tar.gz
rm projects/*/output -rf
rm projects/*/test_data -rf


python scripts/download_test_data.py --model_name coco_funit
output

python inference.py --single_gpu \
--config configs/projects/coco_funit/animal_faces/base64_bs8_class149.yaml \
--output_dir projects/coco_funit/output/animal_faces
output

python inference.py --single_gpu \
--config configs/projects/coco_funit/mammals/base64_bs8_class305.yaml \
--output_dir projects/coco_funit/output/mammals
output

python ./scripts/download_test_data.py --model_name fs_vid2vid
output

python inference.py --single_gpu --num_workers 0 \
--config configs/projects/fs_vid2vid/faceForensics/ampO1.yaml \
--output_dir projects/fs_vid2vid/output/faceForensics
output

python scripts/download_test_data.py --model_name funit
output

python inference.py --single_gpu \
--config configs/projects/funit/animal_faces/base64_bs8_class149.yaml \
--output_dir projects/funit/output/animal_faces
output

python scripts/download_test_data.py --model_name munit
output

python -m torch.distributed.launch --nproc_per_node=1 inference.py \
--config configs/projects/munit/afhq_dog2cat/ampO1.yaml \
--output_dir projects/munit/output/afhq_dog2cat
output

python scripts/download_test_data.py --model_name pix2pixhd
output

python inference.py --single_gpu \
--config configs/projects/pix2pixhd/cityscapes/ampO1.yaml \
--output_dir projects/pix2pixhd/output/cityscapes
output

python scripts/download_test_data.py --model_name spade
output

python -m torch.distributed.launch --nproc_per_node=1 inference.py \
--config configs/projects/spade/cocostuff/base128_bs4.yaml \
--output_dir projects/spade/output/cocostuff
output

python scripts/download_test_data.py --model_name unit
output

python -m torch.distributed.launch --nproc_per_node=1 inference.py \
--config configs/projects/unit/winter2summer/base48_bs1.yaml \
--output_dir projects/unit/output/winter2summer
output

python ./scripts/download_test_data.py --model_name vid2vid
output

python inference.py --single_gpu \
--config configs/projects/vid2vid/cityscapes/ampO1.yaml \
--output_dir projects/vid2vid/output/cityscapes
output

python ./scripts/download_test_data.py --model_name wc_vid2vid
output

python inference.py --single_gpu \
--config configs/projects/wc_vid2vid/cityscapes/seg_ampO1.yaml \
--output_dir projects/wc_vid2vid/output/cityscapes
output