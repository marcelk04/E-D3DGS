export CUDA_VISIBLE_DEVICES=6

#python pre_vci.py -s /data/student_kaempchen/ed3dgs-data/vci/sample_0000 --gaussian_splatting -r 4 --use_masks --camera SIMPLE_PINHOLE -c /data/student_kaempchen/ed3dgs-data/vci/calibration_dome.json
python pre_vci.py -s /data/student_kaempchen/ed3dgs-data/vci/2024_12_12_dynamic3 -c /data/student_kaempchen/ed3dgs-data/vci/2024_12_12_dynamic3/poses.json --remove_background --rotate_images --use_mapper