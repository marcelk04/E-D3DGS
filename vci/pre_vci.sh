export CUDA_VISIBLE_DEVICES=7

python pre_vci.py -s /data/student_kaempchen/ed3dgs-data/vci/sample_0014 --gaussian_splatting -r 4 --use_masks --camera SIMPLE_PINHOLE -c /data/student_kaempchen/ed3dgs-data/vci/calibration_dome.json