export CUDA_VISBLE_DEVICES=6,7

DATA_PATH="/data/student_kaempchen/ed3dgs-data/n3v"
SCENE="cut_roasted_beef"

python script/downsample_point.py $DATA_PATH/$SCENE/colmap/dense/workspace/fused.ply $DATA_PATH/$SCENE/points3D_downsample.ply

