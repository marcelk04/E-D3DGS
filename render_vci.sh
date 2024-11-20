export CUDA_VISIBLE_DEVICES=6,7

SAVE_PATH="output"

DATASET="hypernerf"
SCENE="vrig-chicken"

CONFIG="vrig-chicken"

python render.py --model_path $SAVE_PATH/$DATASET/$CONFIG --configs arguments/$DATASET/$CONFIG.py --skip_train

