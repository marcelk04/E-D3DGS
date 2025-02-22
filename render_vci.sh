export CUDA_VISIBLE_DEVICES=6

SAVE_PATH="output"

DATASET="vci"
SCENE="02_20_01"

CONFIG="default"

python render.py --model_path $SAVE_PATH/$DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py
