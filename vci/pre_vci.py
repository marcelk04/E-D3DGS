import argparse
import os
import glob
import shutil
import numpy as np
import json

# dont question this ... :/
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from script.thirdparty.pre_colmap import *
from script.thirdparty.my_utils import rotmat2qvec

def create_dir(path: str) -> bool:
	if not os.path.exists(path):
		os.makedirs(path)
		return True
	
	return False

def exec_cmd(cmd: str) -> None:
	print(f"Executing '{cmd}'")

	exit_code = os.system(cmd)

	if exit_code != 0:
		exit(exit_code)

	print()

def extract_images(path: str) -> list[str]:
	"""
	Finds all images matching the name C****.* in the directory given by path
	and copies them into the directory colmap/input/.
	This will not copy any images if the directory colmap/input/ already exists.

	\param path the path to the directory containing the images.

	/return a list of all the names of the files that were copied.
	"""

	# TODO: replace file type with wildcard
	images = sorted(glob.glob(os.path.join(path, "[C][0-9][0-9][0-9][0-9].jpg")))
	filenames = [os.path.basename(img) for img in images]

	images_path = os.path.join(path, "colmap", "input")
	if not os.path.exists(images_path):
		os.makedirs(images_path)
	else:
		print(f"'{images_path}' already exists. Skipping image extraction!")
		return filenames
	
	for img in images:
		shutil.copy(img, images_path)

	print(f"Copied {len(images)} images into '{images_path}'")

	return filenames

def extract_poses(path: str, data_type_suffix: str) -> None:
	calibration_path = os.path.join(path, "calibration.json")
	assert os.path.exists(calibration_path)

	calibration_file = open(calibration_path)
	calibration = json.load(calibration_file)

	colmap_path = os.path.join(path, "colmap")
	images_path = os.path.join(colmap_path, "input")
	manual_path = os.path.join(colmap_path, "manual")

	create_dir(manual_path)

	imagestxt_path = os.path.join(manual_path, "images.txt")
	camerastxt_path = os.path.join(manual_path, "cameras.txt")
	points3Dtxt_path = os.path.join(manual_path, "points3D.txt")

	imagetxt_list = []
	cameratxt_list = []

	db_path = os.path.join(colmap_path, "input.db")
	if os.path.exists(db_path):
		print(f"Overwriting old input database at '{db_path}'")
		os.remove(db_path)

	db = COLMAPDatabase.connect(db_path)
	db.create_tables()

	print(f"Start writing to new database at '{db_path}'")

	for i, camera in enumerate(calibration["cameras"]):
		focal_length = 6393.60

		# TODO: why is resolution only 0.5 of the actual resolution?
		width = camera["intrinsics"]["resolution"][0] * 2
		height = camera["intrinsics"]["resolution"][1] * 2

		camera_name = camera["camera_id"]
		image_name = camera_name + data_type_suffix

		view_matrix = np.array(camera["extrinsics"]["view_matrix"]).reshape((4, 4))

		T = view_matrix[:3, 3] # translation vector
		R = view_matrix[:3, :3] # rotation matrix
		Q = rotmat2qvec(R) # rotation quaternion (hopefully)

		# Write camera and image data into database
		params = np.array([focal_length, focal_length, width//2, height//2])
		camera_id = db.add_camera(1, width, height, params)

		image_id = db.add_image(image_name, camera_id, Q, T, image_id=i+1)

		db.commit()

		# Write to cameras.txt and images.txt
		Q_string = " ".join([str(q) for q in Q])
		T_string = " ".join([str(t) for t in T])

		image_line = f"{image_id} {Q_string} {T_string} {camera_id} {image_name}\n"

		imagetxt_list.append(image_line)
		imagetxt_list.append("\n")

		camera_line = f"{camera_id} PINHOLE {str(width)} {str(height)} {str(focal_length)} {str(focal_length)} {str(width//2)} {str(height//2)}\n"
		cameratxt_list.append(camera_line)

	db.close()
	calibration_file.close()

	print("Done writing to database")
	
	with open(imagestxt_path, "w") as f:
		for line in imagetxt_list:
			f.write(line)

	with open(camerastxt_path, "w") as f:
		for line in cameratxt_list:
			f.write(line)

	with open(points3Dtxt_path, "w") as f:
		pass

def run_colmap(path: str) -> None:
	colmap_path = os.path.join(path, "colmap")

	db_path = os.path.join(colmap_path, "input.db")
	image_path = os.path.join(colmap_path, "input")
	manual_path = os.path.join(colmap_path, "manual")

	sparse_path = os.path.join(colmap_path, "distorted", "sparse")
	create_dir(sparse_path)

	dense_path = os.path.join(colmap_path, "dense", "workspace")
	create_dir(dense_path)

	output_path = os.path.join(dense_path, "fused.ply")
	
	# Colmap commands
	print("Starting COLMAP...")
	print()

	feature_extract = f"colmap feature_extractor --database_path {db_path} --image_path {image_path}"
	exec_cmd(feature_extract)

	feature_matcher = f"colmap exhaustive_matcher --database_path {db_path}"
	exec_cmd(feature_matcher)

	tri_and_map = f"colmap point_triangulator --database_path {db_path} --image_path {image_path} --input_path {manual_path} --output_path {sparse_path} --Mapper.ba_global_function_tolerance=0.000001"
	exec_cmd(tri_and_map)

	image_undistortion = f"colmap image_undistorter --image_path {image_path} --input_path {sparse_path} --output_path {dense_path} --output_type COLMAP"
	exec_cmd(image_undistortion)

	patch_match_stereo = f"colmap patch_match_stereo --workspace_path {dense_path}"
	exec_cmd(patch_match_stereo)

	stereo_fusion = f"colmap stereo_fusion --workspace_path {dense_path} --output_path {output_path}"
	exec_cmd(stereo_fusion)

	print(f"All done! The output is in '{output_path}'")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--imagepath", default="", type=str)
	args = parser.parse_args()

	image_path = args.imagepath
	assert os.path.exists(image_path)

	print(f"Preparing data from '{image_path}'")

	images = extract_images(image_path)

	extract_poses(image_path, images[0][-4:])

	run_colmap(image_path)

if __name__ == "__main__":
	main()