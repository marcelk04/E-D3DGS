export CUDA_VISIBLE_DEVICES=6,7

python pre_vci.py -s /data/student_kaempchen/ed3dgs-data/vci/stefan_sample -r 4 --gaussian_splatting -c /data/student_kaempchen/ed3dgs-data/vci/calibration_dome.json --use_masks --camera PINHOLE