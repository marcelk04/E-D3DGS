export CUDA_VISIBLE_DEVICES=7

WORKSPACE="/data/student_kaempchen/ed3dgs-data/vci/test_scene"

python pre_vci_wo_poses.py --workspace $WORKSPACE --imagepath $WORKSPACE/images