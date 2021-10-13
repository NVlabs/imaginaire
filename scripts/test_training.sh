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
BASE_CMD="python -m torch.distributed.launch --nproc_per_node=1 train.py "

CONFIG=configs/unit_test/spade.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/pix2pixHD.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/munit.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/munit_patch.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/unit.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/funit.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/coco_funit.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/biggan.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/stylegan.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/vid2vid_street.yaml
if test -f "$CONFIG"; then
  cmd="python train.py --single_gpu --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/vid2vid_pose.yaml
if test -f "$CONFIG"; then
  cmd="python train.py --single_gpu --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/fs_vid2vid_face.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/fs_vid2vid_pose.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/wc_vid2vid.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/first_order_motion.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/face_vid2vid.yaml
if test -f "$CONFIG"; then
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
fi

CONFIG=configs/unit_test/gancraft.yaml
if test -f "$CONFIG"; then
  python scripts/download_test_data.py --model_name gancraft;
  cmd="${BASE_CMD} --config $CONFIG >> ${LOG} "
  output
  rm -rf projects/gancraft/test_data*
fi