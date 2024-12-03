import argparse
import os
import glob
import shutil
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

# dont question this ... :/
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from script.thirdparty.pre_colmap import *
#from vci.COLMAPDatabase import *
from vci.pose_utils import *
from vci.sys_utils import *

# Helper functions
def load_paths(args: argparse.Namespace) -> dict[str, str]:
	paths = {}
	paths['base'] = args.source_path
	paths['calibration'] = os.path.join(paths['base'], args.calibration_file)

	if args.mask_source == "":
		paths['mask_source'] = paths['base']
	else:
		paths['mask_source'] = args.mask_source

	paths['colmap'] = os.path.join(paths['base'], "colmap")

	paths['masks'] = os.path.join(paths['colmap'], "masks")
	paths['input'] = os.path.join(paths['colmap'], "input")
	paths['distorted'] = os.path.join(paths['colmap'], "distorted")
	paths['manual'] = os.path.join(paths['colmap'], "manual")

	paths['imagestxt'] = os.path.join(paths['manual'], "images.txt")
	paths['camerastxt'] = os.path.join(paths['manual'], "cameras.txt")
	paths['points3Dtxt'] = os.path.join(paths['manual'], "points3D.txt")

	if args.gaussian_splatting:
		paths['db'] = os.path.join(paths['distorted'], "database.db")

		paths['sparse'] = os.path.join(paths['distorted'], "sparse", "0")

		paths['dense'] = paths['colmap']
		paths['output'] = "" # Not used
	else:
		paths['db'] = os.path.join(paths['colmap'], "database.db")
		
		paths['sparse'] = os.path.join(paths['distorted'], "sparse")

		paths['dense'] = os.path.join(paths['colmap'], "dense", "workspace")
		paths['output'] = os.path.join(paths['dense'], "fused.ply")

		
	paths['undistorted_images'] = os.path.join(paths['dense'], "images")

	return paths

# Main functionality
def extract_images(paths: dict[str, str], args: argparse.Namespace) -> list[str]:
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

	s = 1.0 / args.resolution # Downscale factor
	
	for i, img in tqdm(enumerate(images), desc="Scaling and copying images", total=len(filenames)):
		img_loaded = Image.open(img)
		img_loaded = scale_image(img_loaded, s)
		img_loaded.save(os.path.join(paths['input'], filenames[i]))

	print(f"Copied {len(images)} images into '{paths['input']}'")

	return filenames

def extract_masks(paths: dict[str, str], filenames: list[str], args: argparse.Namespace) -> None:
	if len(filenames) == 0:
		print("Found no files. Skipping mask extraction!")
		return

	masks = [file[:-4] + "_mask" for file in filenames]

	if not create_dir(paths['masks']):
		print(f"'{paths['masks']}' already exists. Skipping mask extraction!")
		return

	s = 1.0 / args.resolution
	
	for i, file in tqdm(enumerate(filenames), desc="Scaling and copying masks", total=len(filenames)):
		src_path = os.path.join(paths['mask_source'], file[:-4] + "_mask.png")
		dst_path = os.path.join(paths['masks'], file[:-4] + ".jpg.png")

		mask_loaded = Image.open(src_path)
		mask_loaded = scale_image(mask_loaded, s)
		mask_loaded.save(dst_path)

	print(f"Copied {len(masks)} masks into '{paths['masks']}'")

def extract_poses(paths: dict[str, str], args: argparse.Namespace, data_type_suffix: str) -> None:
	# Set camera model
	camera_models = {"PINHOLE": 1 , "OPENCV": 4 }
	if not args.camera in camera_models.keys():
		print(f"Unsupported camera model: {args.camera}")
		exit()

	camera_model = camera_models[args.camera]

	# Load calibration file
	assert os.path.exists(paths['calibration'])
	calibration_file = open(paths['calibration'])
	calibration = json.load(calibration_file)
	calibration_file.close()

	create_dir(paths['manual'])
	create_dir(paths['distorted'])

	imagetxt_list = []
	cameratxt_list = []

	# Delete old database
	if os.path.exists(paths['db']):
		print(f"Overwriting old input database at '{paths['db']}'")
		os.remove(paths['db'])

	db = COLMAPDatabase.connect(paths['db'])
	db.create_tables()

	print(f"Start writing to new database at '{paths['db']}'")

	for i, camera in tqdm(enumerate(calibration["cameras"]), desc="Reading camera calibration", total=len(calibration['cameras'])):
		# Downscale factor
		s = 1.0 / args.resolution

		# Extract pose information from the JSON file

		camera_name = camera["camera_id"]
		image_name = camera_name + data_type_suffix
		
		view_matrix = np.array(camera["extrinsics"]["view_matrix"]).reshape((4, 4))
		view_matrix = rotate_z_view_matrix(view_matrix)

		camera_matrix = np.array(camera["intrinsics"]["camera_matrix"]).reshape((3, 3))

		width = camera["intrinsics"]["resolution"][0]
		height = camera["intrinsics"]["resolution"][1]

		distortion_coefficients = np.array(camera["intrinsics"]["distortion_coefficients"])

		# Camera rotation and translation
		R = view_matrix[:3, :3]
		T = view_matrix[:3, 3]
		Q = rotation_matrix_to_quaternion(R)

		# focal length
		f_x = camera_matrix[0, 0]
		f_y = camera_matrix[1, 1]

		# principal point
		c_x = camera_matrix[0, 2]
		c_y = camera_matrix[1, 2]

		# Camera C1004 (id 31) has different dimensions for some reason...
		if width != 5328:
			width *= 2
			height *= 2
			f_x *= 2
			f_y *= 2
			c_x *= 2
			c_y *= 2

		# Downscale
		width *= s
		height *= s

		f_x *= s
		f_y *= s

		c_x *= s
		c_y *= s

		# Write camera and image data into database
		if camera_model == 1: # PINHOLE
			params = np.array([f_x, f_y, c_x, c_y])
		else: # OPENCV
			params = np.array([f_x, f_y, c_x, c_y, distortion_coefficients[0], distortion_coefficients[1], distortion_coefficients[2], distortion_coefficients[3]])

		camera_id = db.add_camera(camera_model, width, height, params)

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

		camera_line = f"{camera_id} {args.camera} {width} {height} {params_string}\n"
		cameratxt_list.append(camera_line)

	db.close()

	print("Done writing to database")
	
	# Write prepared data into images.txt and cameras.txt
	write_lines_to_file(imagetxt_list, paths['imagestxt'])
	write_lines_to_file(cameratxt_list, paths['camerastxt'])
	write_lines_to_file([], paths['points3Dtxt'])

def run_colmap(paths: dict[str, str], args: argparse.Namespace, image_names: list[str]) -> None:
	create_dir(paths['sparse'])
	create_dir(paths['dense'])

	# Clear colmap/dense/images
	removed_images = remove_files(paths['undistorted_images'], image_names)
	print(f"Removed {removed_images} images from '{paths['undistorted_images']}'")
	
	# Colmap commands
	print("Starting COLMAP...")
	print()

	feature_extract = f"colmap feature_extractor \
		--database_path {paths['db']} \
		--image_path {paths['input']}"
	if args.use_masks:
		feature_extract += f" --ImageReader.mask_path {paths['masks']}"
	exec_cmd(feature_extract)

	feature_matcher = f"colmap exhaustive_matcher \
		--database_path {paths['db']}" # --TwoViewGeometry.min_num_inliers 5
	exec_cmd(feature_matcher)

	tri_and_map = f"colmap point_triangulator \
		--database_path {paths['db']} \
		--image_path {paths['input']} \
		--input_path {paths['manual']} \
		--output_path {paths['sparse']}" # --Mapper.min_num_matches 5 --Mapper.init_min_num_inliers 40 --Mapper.ba_global_function_tolerance=0.000001
	exec_cmd(tri_and_map)

	image_undistortion = f"colmap image_undistorter \
		--image_path {paths['input']} \
		--input_path {paths['sparse']} \
		--output_path {paths['dense']}"
	exec_cmd(image_undistortion)

	if not args.gaussian_splatting:
		patch_match_stereo = f"colmap patch_match_stereo \
			--workspace_path {paths['dense']}"
		exec_cmd(patch_match_stereo)

		stereo_fusion = f"colmap stereo_fusion \
			--workspace_path {paths['dense']} \
			--output_path {paths['output']}" # --StereoFusion.mask_path {mask_path}
		exec_cmd(stereo_fusion)

		print(f"All done! The output is in '{paths['output']}'")
	else:
		sparse = os.path.join(paths['dense'], "sparse")
		sparse0 = os.path.join(paths['dense'], "sparse", "0")

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
	parser = argparse.ArgumentParser(prog="python pre_vci.py", description="Generates a sparse/dense point cloud from a set of input images and camera poses generated by the VCI")
	parser.add_argument("--source_path", "-s", default="", type=str, help="the path where the image files are located")
	parser.add_argument("--mask_source", default="", type=str, help="the path to the image masks (default: source_path)")
	parser.add_argument("--resolution", "-r", default=1, type=int, choices=[1,2,4,8])
	parser.add_argument("--calibration_file", "-c", default="calibration.json", type=str, help="the name of the camera calibration file (default: %(default)s)")
	parser.add_argument("--gaussian_splatting", action="store_true", help="enables output for 3d gaussian splatting")
	parser.add_argument("--use_masks", action="store_true", help="enables usage of masks in the feature extractor")
	parser.add_argument("--camera", default="OPENCV", type=str, choices=["OPENCV", "PINHOLE"], help="the camera model used when extracting the poses (default: %(default)s)")
	args = parser.parse_args()

	# Load all the paths from the passed arguments
	paths = load_paths(args)
	assert os.path.exists(paths['base'])

	# Print out the paths
	print("Set directories:")
	for k, v in paths.items():
		print(f"{k:<20} -> {v}")
	print()

	print(f"Preparing data from '{paths['base']}'")

	images = extract_images(paths, args)

	if args.use_masks:
		extract_masks(paths, images, args)

	extract_poses(paths, args, images[0][-4:])

	run_colmap(paths, args, images)

if __name__ == "__main__":
	main()