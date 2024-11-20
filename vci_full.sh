export CUDA_VISIBLE_DEVICES=6,7

GT_PATH="/data/student_kaempchen/ed3dgs-data/n3v"
SAVE_PATH="output"

DATASET="dynerf"
SCENE="cut_roasted_beef"

CONFIG="cut_roasted_beef"

python train.py -s $GT_PATH/$SCENE --model_path $SAVE_PATH/$DATASET/$CONFIG --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py -r 1

python render.py --model_path $SAVE_PATH/$DATASET/$CONFIG --skip_train --configs arguments/$DATASET/$CONFIG.py

python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG
