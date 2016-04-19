#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

#case $DATASET in
#  pascal_voc)
 #   TRAIN_IMDB="voc_2007_trainval"
  #  TEST_IMDB="voc_2007_test"
   # PT_DIR="pascal_voc"
   # ITERS=70000
   # ;;
  #coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
   # TRAIN_IMDB="coco_2014_train"
   # TEST_IMDB="coco_2014_minival"
   # PT_DIR="coco"
   # ITERS=490000
   # ;;
  #*)
   # echo "No dataset given"
    #exit
   # ;;
#esac
TRAIN_IMDB="hover_train"
TEST_IMDB="hover_test"
PT_DIR="hover"
ITERS=40000

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

 #--weights /home/will/dev/py-faster-rcnn/models/hover/places205VGG16/snapshot_iter_765280.caffemodel \
  #--weights /home/will/dev/py-faster-rcnn/models/hover/places205VGG16/snapshot_iter_765280.caffemodel \
 #--weights data/imagenet_models/${NET}.v2.caffemodel \
  #--weights data/imagenet_models/${NET}.v2.caffemodel \

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net /home/will/dev/py-faster-rcnn/output/faster_rcnn_end2end/train/vgg16_faster_rcnn_end2end_p70_iter_40000.caffemodel  \
  --imdb hover_test \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
