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
#from vci.COLMAPDatabase import *
from script.thirdparty.my_utils import rotmat2qvec

def load_paths(args: argparse.Namespace) -> dict[str, str]:
	paths = {}
	paths['base'] = args.source_path
	paths['calibration'] = os.path.join(paths['base'], args.calibration_file)

	paths['colmap'] = os.path.join(paths['base'], 'colmap')

	paths['masks'] = os.path.join(paths['colmap'], 'masks')
	paths['input'] = os.path.join(paths['colmap'], 'input')
	paths['distorted'] = os.path.join(paths['colmap'], 'distorted')
	paths['manual'] = os.path.join(paths['colmap'], 'manual')

	paths['imagestxt'] = os.path.join(paths['manual'], "images.txt")
	paths['camerastxt'] = os.path.join(paths['manual'], "cameras.txt")
	paths['points3Dtxt'] = os.path.join(paths['manual'], "points3D.txt")

	if args.gaussian_splatting:
		paths['db'] = os.path.join(paths['distorted'], "database.db")

		paths['sparse'] = os.path.join(paths['distorted'], 'sparse', "0")

		paths['dense'] = paths['colmap']
		paths['output'] = "" # Not used
	else:
		paths['db'] = os.path.join(paths['colmap'], "database.db")
		
		paths['sparse'] = os.path.join(paths['distorted'], 'sparse')

		paths['dense'] = os.path.join(paths['colmap'], 'dense', "workspace")
		paths['output'] = os.path.join(paths['dense'], "fused.ply")

		
	paths['undistorted_images'] = os.path.join(paths['dense'], "images")

	return paths
	

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

def extract_images(paths: dict[str, str]) -> list[str]:
	"""
	Finds all images matching the name C****.* in the directory given by path
	and copies them into the directory colmap/input/.
	This will not copy any images if the directory colmap/input/ already exists.
	"""

	images = sorted(glob.glob(os.path.join(paths['base'], "[C][0-9][0-9][0-9][0-9].*")))
	filenames = [os.path.basename(img) for img in images]

	if not create_dir(paths['input']):
		print(f"'{paths['input']}' already exists. Skipping image extraction!")
		return filenames
	
	for img in images:
		shutil.copy(img, paths['input'])

	print(f"Copied {len(images)} images into '{paths['input']}'")

	return filenames

def extract_masks(paths: dict[str, str], filenames: list[str]) -> None:
	if len(filenames) == 0:
		print("Found no files. Skipping mask extraction!")
		return

	file_type = ".png"
	masks = [file[:-4] + "_mask" for file in filenames]

	if not create_dir(paths['masks']):
		print(f"'{paths['masks']}' already exists. Skipping mask extraction!")
		return
	
	for file in filenames:
		src_path = os.path.join(paths['base'], file[:-4] + "_mask.png")
		dst_path = os.path.join(paths['masks'], file[:-4] + ".jpg.png")
		shutil.copy(src_path, dst_path)

	print(f"Copied {len(masks)} masks into '{paths['masks']}'")

def extract_poses(paths: dict[str, str], data_type_suffix: str) -> None:
	assert os.path.exists(paths['calibration'])

	calibration_file = open(paths['calibration'])
	calibration = json.load(calibration_file)

	create_dir(paths['manual'])
	create_dir(paths['distorted'])

	imagetxt_list = []
	cameratxt_list = []

	if os.path.exists(paths['db']):
		print(f"Overwriting old input database at '{paths['db']}'")
		os.remove(paths['db'])

	db = COLMAPDatabase.connect(paths['db'])
	db.create_tables()

	print(f"Start writing to new database at '{paths['db']}'")

	for i, camera in enumerate(calibration["cameras"]):
		# Extract information about the poses from the JSON file

		# TODO: why is resolution only 0.5 of the actual resolution?
		width = camera["intrinsics"]["resolution"][0] * 2
		height = camera["intrinsics"]["resolution"][1] * 2

		camera_name = camera["camera_id"]
		image_name = camera_name + data_type_suffix

		view_matrix = np.array(camera["extrinsics"]["view_matrix"]).reshape((4, 4))

		T = view_matrix[:3, 3] # translation vector
		R = view_matrix[:3, :3] # rotation matrix
		Q = rotmat2qvec(R) # rotation quaternion (hopefully)

		camera_matrix = np.array(camera["intrinsics"]["camera_matrix"]).reshape((3, 3))

		focal_length_x = camera_matrix[0, 0]
		focal_length_y = camera_matrix[1, 1]

		# TODO: again only 0.5x?
		principal_point_x = camera_matrix[0, 2] * 2
		principal_point_y = camera_matrix[1, 2] * 2

		distortion_coefficients = np.array(camera["intrinsics"]["distortion_coefficients"])

		# Write camera and image data into database
		params = np.array([focal_length_x, focal_length_y, principal_point_x, principal_point_y, distortion_coefficients[0], distortion_coefficients[1], distortion_coefficients[2], distortion_coefficients[3]])

		# Camera Models:
		# PINHOLE = 1
		# RADIAL = 3
		# OPENCV = 4
		camera_id = db.add_camera(4, width, height, params)

		image_id = db.add_image(image_name, camera_id, Q, T, image_id=i+1)
		#image_id = db.add_image(image_name, camera_id, image_id=i+1)

		db.commit()

		# Append lines for images.txt and cameras.txt
		Q_string = " ".join([str(q) for q in Q])
		T_string = " ".join([str(t) for t in T])
		params_string = " ".join([str(num) for num in params])

		#camera_id = i+1
		#image_id = camera_id

		image_line = f"{image_id} {Q_string} {T_string} {camera_id} {image_name}\n"

		imagetxt_list.append(image_line)
		imagetxt_list.append("\n")

		camera_line = f"{camera_id} OPENCV {width} {height} {params_string}\n"
		cameratxt_list.append(camera_line)

	db.close()
	calibration_file.close()

	print("Done writing to database")
	
	# Write prepared data into images.txt and cameras.txt
	with open(paths['imagestxt'], "w") as f:
		for line in imagetxt_list:
			f.write(line)

	with open(paths['camerastxt'], "w") as f:
		for line in cameratxt_list:
			f.write(line)

	with open(paths['points3Dtxt'], "w") as f:
		pass

def remove_undistorted_images(colmap_dense_path: str, filenames: list[str]) -> int:
	count = 0

	for file in filenames:
		filepath = os.path.join(colmap_dense_path, "images", file)

		if os.path.exists(filepath):
			os.remove(filepath)
			count += 1

	return count

def run_colmap(paths: dict[str, str], args: argparse.Namespace, image_names: list[str]) -> None:
	create_dir(paths['sparse'])
	create_dir(paths['dense'])

	# Clear colmap/dense/images
	removed_images = remove_undistorted_images(paths['undistorted_images'], image_names)
	print(f"Removed {removed_images} images from '{paths['undistorted_images']}'")
	
	# Colmap commands
	print("Starting COLMAP...")
	print()

	feature_extract = f"colmap feature_extractor --database_path {paths['db']} --image_path {paths['input']}"
	if args.use_masks:
		feature_extract += f" --ImageReader.mask_path {paths['masks']}"
	exec_cmd(feature_extract)

	feature_matcher = f"colmap exhaustive_matcher --database_path {paths['db']}" # --TwoViewGeometry.min_num_inliers 5
	exec_cmd(feature_matcher)

	tri_and_map = f"colmap point_triangulator --database_path {paths['db']} --image_path {paths['input']} --input_path {paths['manual']} --output_path {paths['sparse']} --Mapper.ba_global_function_tolerance=0.000001" # --Mapper.min_num_matches 5 --Mapper.init_min_num_inliers 40 --Mapper.ba_global_function_tolerance=0.000001
	exec_cmd(tri_and_map)

	image_undistortion = f"colmap image_undistorter --image_path {paths['input']} --input_path {paths['sparse']} --output_path {paths['dense']}"
	exec_cmd(image_undistortion)

	if not args.gaussian_splatting:
		patch_match_stereo = f"colmap patch_match_stereo --workspace_path {paths['dense']}"
		exec_cmd(patch_match_stereo)

		stereo_fusion = f"colmap stereo_fusion --workspace_path {paths['dense']} --output_path {paths['output']}" # --StereoFusion.mask_path {mask_path}
		exec_cmd(stereo_fusion)

		print(f"All done! The output is in '{paths['output']}'")
	else:
		sparse = os.path.join(paths['dense'], 'sparse')
		sparse0 = os.path.join(paths['dense'], 'sparse', "0")

		files = os.listdir(sparse)
		create_dir(sparse0)

		# Copy each file from the source directory to the destination directory
		for file in files:
			if file == '0':
				continue
			source_file = os.path.join(sparse, file)
			destination_file = os.path.join(sparse0, file)
			shutil.move(source_file, destination_file)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path", "-s", default="", type=str)
	parser.add_argument("--calibration_file", "-c", default="calibration.json", type=str)
	parser.add_argument("--gaussian_splatting", action="store_true")
	parser.add_argument("--use_masks", action="store_true")
	args = parser.parse_args()

	paths = load_paths(args)

	for k, v in paths.items():
		print(f"{k:<20} -> {v}")

	assert os.path.exists(paths['base'])

	print(f"Preparing data from '{paths['base']}'")

	images = extract_images(paths)

	if args.use_masks:
		extract_masks(paths, images)

	extract_poses(paths, images[0][-4:])

	run_colmap(paths, args, images)

if __name__ == "__main__":
	main()