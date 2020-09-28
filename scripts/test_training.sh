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
#BASE_CMD="python train.py --single_gpu "
BASE_CMD="python -m torch.distributed.launch --nproc_per_node=1 train.py "

# Paired image translation
cmd="python scripts/build_lmdb.py --config configs/unit_test/spade.yaml \
--paired --data_root dataset/unit_test/raw/spade/ --output_root dataset/unit_test/lmdb/spade --overwrite >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/spade.yaml >> ${LOG} "
output

cmd="python scripts/build_lmdb.py --config configs/unit_test/pix2pixHD.yaml \
--paired --data_root dataset/unit_test/raw/pix2pixHD/ --output_root dataset/unit_test/lmdb/pix2pixHD --overwrite >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/pix2pixHD.yaml >> ${LOG} "
output

# Unpaired translation
cmd="python scripts/build_lmdb.py --config configs/unit_test/munit.yaml \
--data_root dataset/unit_test/raw/munit/ --output_root dataset/unit_test/lmdb/munit --overwrite >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/munit.yaml >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/munit_patch.yaml >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/unit.yaml >> ${LOG} "
output

# Example-guided translation
cmd="python scripts/build_lmdb.py --config configs/unit_test/funit.yaml \
--data_root dataset/unit_test/raw/funit/ --output_root dataset/unit_test/lmdb/funit --overwrite >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/funit.yaml >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/coco_funit.yaml >> ${LOG} "
output

# vid2vid
cmd="python scripts/build_lmdb.py --config configs/unit_test/vid2vid_street.yaml \
--paired --data_root dataset/unit_test/raw/vid2vid/street/ --output_root dataset/unit_test/lmdb/vid2vid/street --overwrite >> ${LOG} "
output

cmd="python train.py --single_gpu --config configs/unit_test/vid2vid_street.yaml >> ${LOG} "
output

cmd="python scripts/build_lmdb.py --config configs/unit_test/vid2vid_pose.yaml \
--paired --data_root dataset/unit_test/raw/vid2vid/pose/ --output_root dataset/unit_test/lmdb/vid2vid/pose --overwrite >> ${LOG} "
output

cmd="python train.py --single_gpu --config configs/unit_test/vid2vid_pose.yaml >> ${LOG} "
output

# fs-vid2vid
cmd="python scripts/build_lmdb.py --config configs/unit_test/fs_vid2vid_face.yaml \
--paired --data_root dataset/unit_test/raw/fs_vid2vid/face/ --output_root dataset/unit_test/lmdb/fs_vid2vid/face --overwrite >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/fs_vid2vid_face.yaml >> ${LOG} "
output

cmd="python scripts/build_lmdb.py --config configs/unit_test/fs_vid2vid_pose.yaml \
--paired --data_root dataset/unit_test/raw/vid2vid/pose/ --output_root dataset/unit_test/lmdb/fs_vid2vid/pose --overwrite >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/fs_vid2vid_pose.yaml >> ${LOG} "
output

# wc_vid2vid
cmd="python scripts/build_lmdb.py --config configs/unit_test/wc_vid2vid.yaml \
--paired --data_root dataset/unit_test/raw/wc_vid2vid/cityscapes/ --output_root dataset/unit_test/lmdb/wc_vid2vid/cityscapes --overwrite >> ${LOG} "
output

cmd="${BASE_CMD} --config configs/unit_test/wc_vid2vid.yaml >> ${LOG} "
output
