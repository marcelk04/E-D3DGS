export CUDA_VISIBLE_DEVICES=6,7

SCENE="vrig-broom"

python script/pre_hypernerf.py --videopath /data/student_kaempchen/ed3dgs-data/hypernerf/$SCENE
