export CUDA_VISIBLE_DEVICES=6

SAVE_PATH="output"

DATASET="vci"
SCENE="03_09_01_batch_size"

CONFIG="2024_12_12_dynamic3"

python render.py --model_path $SAVE_PATH/$DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py --save_images
python metrics.py -m $SAVE_PATH/$DATASET/$SCENE
