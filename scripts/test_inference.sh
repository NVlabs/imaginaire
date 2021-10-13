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

LOG="/tmp/unit_test.log"

rm projects/*/*.tar.gz
rm projects/*/test_data -rf

cmd="python scripts/download_test_data.py --model_name coco_funit"
output

CONFIG=configs/projects/coco_funit/animal_faces/base64_bs8_class149.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu \
  --config $CONFIG \
  --output_dir projects/coco_funit/output/animal_faces"
  output
fi

CONFIG=configs/projects/coco_funit/mammals/base64_bs8_class305.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu \
  --config $CONFIG \
  --output_dir projects/coco_funit/output/mammals"
  output
fi

cmd="python ./scripts/download_test_data.py --model_name fs_vid2vid"
output

CONFIG=configs/projects/fs_vid2vid/face_forensics/ampO1.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu --num_workers 0 \
  --config $CONFIG \
  --output_dir projects/fs_vid2vid/output/face_forensics"
  output
fi

cmd="python scripts/download_test_data.py --model_name funit"
output

CONFIG=configs/projects/funit/animal_faces/base64_bs8_class149.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu \
  --config $CONFIG \
  --output_dir projects/funit/output/animal_faces"
  output
fi

cmd="python scripts/download_test_data.py --model_name munit"
output

CONFIG=configs/projects/munit/afhq_dog2cat/ampO1.yaml
if test -f "$CONFIG"; then
  cmd="python -m torch.distributed.launch --nproc_per_node=1 inference.py \
  --config $CONFIG \
  --output_dir projects/munit/output/afhq_dog2cat"
  output
fi

cmd="python scripts/download_test_data.py --model_name pix2pixhd"
output

CONFIG=configs/projects/pix2pixhd/cityscapes/ampO1.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu \
  --config $CONFIG \
  --output_dir projects/pix2pixhd/output/cityscapes"
  output
fi

cmd="python scripts/download_test_data.py --model_name spade"
output

CONFIG=configs/projects/spade/cocostuff/base128_bs4.yaml
if test -f "$CONFIG"; then
  cmd="python -m torch.distributed.launch --nproc_per_node=1 inference.py \
  --config $CONFIG \
  --output_dir projects/spade/output/cocostuff"
  output
fi

cmd="python scripts/download_test_data.py --model_name unit"
output

CONFIG=configs/projects/unit/winter2summer/base48_bs1.yaml
if test -f "$CONFIG"; then
  cmd="python -m torch.distributed.launch --nproc_per_node=1 inference.py \
  --config $CONFIG \
  --output_dir projects/unit/output/winter2summer"
  output
fi

cmd="python ./scripts/download_test_data.py --model_name vid2vid"
output

CONFIG=configs/projects/vid2vid/cityscapes/ampO1.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu \
  --config $CONFIG \
  --output_dir projects/vid2vid/output/cityscapes"
  output
fi

cmd="python ./scripts/download_test_data.py --model_name \"wc_vid2vid/cityscapes\""
output

CONFIG=configs/projects/wc_vid2vid/cityscapes/seg_ampO1.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu \
  --config $CONFIG \
  --output_dir projects/wc_vid2vid/output/cityscapes"
  output
fi

cmd="python ./scripts/download_test_data.py --model_name \"wc_vid2vid/mannequin\""
output

CONFIG=configs/projects/wc_vid2vid/mannequin/hed_ampO0.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu \
  --config $CONFIG \
  --output_dir projects/wc_vid2vid/output/mannequin"
  output
fi


cmd="python ./scripts/download_test_data.py --model_name gancraft"
output

CONFIG=configs/projects/gancraft/demoworld.yaml
if test -f "$CONFIG"; then
  cmd="python inference.py --single_gpu \
  --config $CONFIG \
  --output_dir projects/gancraft/output/demoworld"
  output
fi