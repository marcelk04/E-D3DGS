export CUDA_VISIBLE_DEVICES=7

SAVE_PATH="output"

DATASET="vci"
SCENE="01_25_01"

CONFIG="default"

python render.py --model_path $SAVE_PATH/$DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py

