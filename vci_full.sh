export CUDA_VISIBLE_DEVICES=6,7

GT_PATH="/data/student_kaempchen/ed3dgs-data/vci"
SAVE_PATH="output"

DATASET="vci"
SCENE="2024_12_12_dynamic3"
NAME="01_25_01"

CONFIG="default"

python train.py -s $GT_PATH/$SCENE --model_path $SAVE_PATH/$DATASET/$NAME --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py -r 1

python render.py --model_path $SAVE_PATH/$DATASET/$NAME --skip_train --configs arguments/$DATASET/$CONFIG.py

#python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG
