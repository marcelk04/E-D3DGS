import argparse
import os
import glob
import shutil
import json
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

import numpy as np
from skimage import transform
import skimage as ski

# dont question this ... :/
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

#from script.thirdparty.colmap_database_3_9 import *
from script.thirdparty.colmap_database_3_10 import *
from script.downsample_point import process_ply_file
from utils.pose_utils import rotmat2qvec
from vci.pose_utils import *
from vci.sys_utils import *
from vci.mask_utils import *
from vci.colmap_utils import run_colmap_poses, run_colmap_mapper

from submodules.BackgroundMattingV2.model.model import MattingRefine

# Helper functions
def load_paths(args: argparse.Namespace) -> dict[str, str]:
	paths: dict[str, str] = {}

	paths['base'] = args.source_path
	paths['calibration'] = os.path.join(paths['base'], args.calibration_file) # Will discard paths['base'] if args.calibration_file is an absolute path

	paths['bg_src'] = os.path.join(paths['base'], "background", "rgb")

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
	else:
		paths['db'] = os.path.join(paths['colmap'], "database.db")
		
		paths['sparse'] = os.path.join(paths['distorted'], "sparse")
		paths['dense'] = os.path.join(paths['colmap'], "dense", "workspace")

	paths['undistorted_images'] = os.path.join(paths['dense'], "images")
	paths['output'] = os.path.join(paths['dense'], "fused.ply")
	paths['output_downsample'] = os.path.join(paths['base'], "points3D_downsample.ply")

	return paths

def cam_name(idx: int) -> str:
	return "cam" + str(idx).zfill(2)

def rotate_image(image, rotation):
	if rotation == 1:
		return image.transpose(Image.ROTATE_90)
	elif rotation == 2:
		return image.transpose(Image.ROTATE_180)
	elif rotation == 3:
		return image.transpose(Image.ROTATE_270)
	else:
		return image

def mask_image(img_src, bg_src, model):
	mask, img = model(img_src, bg_src)[:2]
	img = img * mask

	masked = torch.concatenate((img[0], mask[0]))

	return to_pil_image(masked.cpu())

def prepare_input_images_ed3dgs(paths: dict[str, str], args: argparse.Namespace) -> tuple[int, int]:
	"""
	"""

	frames = sorted([f for f in os.listdir(paths['base']) if f.startswith("frame")]) # Directory names (frame_00000, frame_00001, ...)
	cameras = sorted([f for f in os.listdir(os.path.join(paths['base'], frames[0], "rgb"))]) # File names (C0000.jpg, C0001.jpg, ...)

	num_frames = len(frames)
	num_cams = len(cameras)

	print(f"Number of frames: {num_frames}")
	print(f"Number of cameras: {num_cams}")
	print(f"Total number of images: {num_frames * num_cams}")
	print()

	# Read background images
	#bgs = ski.io.imread_collection(os.path.join(paths['bg_src'], "*"), conserve_memory=False)
	bg_paths = [os.path.join(paths['bg_src'], bg) for bg in sorted(os.listdir(paths['bg_src']))]
	bgs = [to_tensor(Image.open(path)).cuda().unsqueeze(0) for path in bg_paths]

	# Create image dir
	if not create_dir(os.path.join(paths['base'], "images")):
		print("E-D3DGS input directory already exists. Skipping image extraction.")
		return num_frames, num_cams
	
	# Create cam dirs (images/cam00, images/cam01, ...)
	for j, cam in enumerate(cameras):
		create_dir(os.path.join(paths['base'], "images", cam_name(j)))

	if not args.remove_background:
		# Simply copy the images to the images folder
		progress_bar = tqdm(desc="Copying images", total=num_frames*num_cams)

		for i, frame in enumerate(frames):
			for j, cam in enumerate(cameras):
				src = os.path.join(paths['base'], frame, "rgb", cam)
				dst = os.path.join(paths['base'], "images", cam_name(j), str(i).zfill(4) + ".jpg")

				shutil.copyfile(src, dst)

				progress_bar.update()
		progress_bar.close()

		return num_frames, num_cams
	
	# Load model for removing the backgrounds
	model = MattingRefine(backbone="mobilenetv2", backbone_scale=0.25, refine_mode="sampling", refine_sample_pixels=80000)

	model.load_state_dict(torch.load("pytorch_mobilenetv2.pth", weights_only=True))
	model = model.eval().to(torch.float32).to(torch.device("cuda"))
	
	rotations = [3,1,3,3,3,3,3,1,3,3,3,3,1,1,1,3,3,3,1,3,1,3,1,1,1,1,1]

	# Apply masks and save images in images folder
	progress_bar = tqdm(desc="Removing background", total=num_frames*num_cams)

	for i, frame in enumerate(frames):
		frame_path = os.path.join(paths['base'], frame, "rgb")

		for j, cam in enumerate(cameras):
			progress_bar.set_postfix({ "Cam": os.path.splitext(cam)[0], "Img": i })

			frame_img = to_tensor(Image.open(os.path.join(frame_path, cam))).cuda().unsqueeze(0)
			masked = mask_image(frame_img, bgs[j], model)
			masked = rotate_image(masked, rotations[j])

			dst_path = os.path.join(paths['base'], "images", cam_name(j), str(i).zfill(4) + ".png")
			masked.save(dst_path)

			# # Load input image
			# frame_img = ski.io.imread(os.path.join(frame_path, cam)) / 255.0

			# # Calculate mask and store it in alpha channel
			# masked, mask = calculate_mask(frame_img, bgs[j] / 255.0, model)
			# masked = np.concatenate((masked, mask[..., None]), axis=-1)

			# masked = np.rot90(masked, k=rotations[j], axes=(0,1))

			# # Save as png to keep alpha channel
			# dst_path = os.path.join(paths['base'], "images", cam_name(j), str(i).zfill(4) + ".png")
			# ski.io.imsave(dst_path, np.uint8(masked * 255.0), check_contrast=False)

			progress_bar.update()

	progress_bar.close()

	return num_frames, num_cams


# Main functionality
def prepare_input_images_colmap(paths: dict[str, str], args: argparse.Namespace) -> tuple[list[str], list[str]]:
	"""
	Extract the images and masks provided in paths['base'] and paths['mask_source'] respectively. The
	files will be copied into paths['input'] and paths['masks'] respectively. If --resolution is set
	the files will be downscaled accordingly and if --use_masks is set the masks will be applied.

	Params:
		paths (dict[str, str]):    A dictionary containing the filepaths
		args (argparse.Namespace): The command line arguments generated by argparse

	Returns:
		images (list[str]): A list of all input images found in paths['base']
		masks (list[str]):  A list of all masks found in paths['mask_source']
	"""

	rotations = [3,1,3,3,3,3,3,1,3,3,3,3,1,1,1,3,3,3,1,3,1,3,1,1,1,1,1]

	valid_filetypes = [".png", ".jpg"]

	img_path = os.path.join(paths['base'], "frame_00000", "rgb")
	images = sorted([file for file in os.listdir(img_path) if os.path.splitext(file)[1] in valid_filetypes])
	bgs = sorted([file for file in os.listdir(paths['bg_src']) if os.path.splitext(file)[1] in valid_filetypes])

	if len(images) == 0:
		print("Found no images. Skipping image extraction")
		return (images, bgs)
	
	copy_images = create_dir(paths['input'])
	copy_masks = create_dir(paths['masks']) and args.remove_background

	if not copy_images and not copy_masks:
		print("Image and mask directories already exist. Skipping image extraction (if this is undesired, use --replace_images)")
		return (images, bgs)

	downscale_factor = 1.0 / args.resolution

	model = MattingRefine(backbone="mobilenetv2", backbone_scale=0.25, refine_mode="sampling", refine_sample_pixels=80000)

	model.load_state_dict(torch.load("pytorch_mobilenetv2.pth", weights_only=True))
	model = model.eval().to(torch.float32).to(torch.device("cuda"))

	for i, img_name in tqdm(enumerate(images), "Preparing input images", total=len(images)):
		img_src = ski.io.imread(os.path.join(img_path, img_name)) / 255.0
		img_src = np.rot90(img_src, k=rotations[i], axes=(0,1))

		if args.remove_background:
			bg_src = ski.io.imread(os.path.join(paths['bg_src'], img_name)) / 255.0
			bg_src = np.rot90(bg_src, k=rotations[i], axes=(0,1))

			_, mask = calculate_mask(img_src.copy(), bg_src.copy(), model)
			
			if args.resolution != 1:
				mask = transform.rescale(mask, downscale_factor, anti_aliasing=False)

			ski.io.imsave(os.path.join(paths['masks'], cam_name(i) + ".jpg.png"), np.uint8(mask * 255.0), check_contrast=False)

		if args.resolution != 1:
			img_src = transform.rescale(img_src, downscale_factor, anti_aliasing=False)

		ski.io.imsave(os.path.join(paths['input'], cam_name(i) + ".jpg"), np.uint8(img_src * 255.0), check_contrast=False)

	return images, bgs

def extract_poses(paths: dict[str, str], args: argparse.Namespace) -> None:
	"""
	Extracts camera pose information from paths['calibration']. The poses will be written 
	to the sqlite database located at paths['db'] and to the .txt files located in the directory
	paths['manual']. The generated files can be processed directly by COLMAP.
	
	Params:
		filenames (list[str]):     A list of the filenames of the image sources as returned by extract_images
		paths (dict[str, str]):    A dictionary containing the filepaths
		args (argparse.Namespace): The command line arguments generated by argparse

	Returns:
		None
	"""

	filenames = os.listdir(paths['input'])

	# Set camera model
	camera_models = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3, "OPENCV": 4, "RADIAL_FISHEYE": 9 }
	camera_model = camera_models[args.camera]
	
	# Downscale factor
	s = 1.0 / args.resolution

	# Load calibration file
	assert os.path.exists(paths['calibration'])
	calibration_file = open(paths['calibration'])
	calibration = json.load(calibration_file)
	calibration_file.close()

	# Create necessary directories
	db_dir = os.path.dirname(paths['db'])
	create_dir(db_dir)
	create_dir(paths['manual'])

	# Calculate camera center to shift camera positions to the origin
	avg_P = np.average(np.column_stack([-V[:3, :3].T @ V[:3, 3] for V in [np.array(cam["extrinsics"]["view_matrix"]).reshape((4, 4)) for cam in calibration["cameras"]]]), axis=1)
	print(f"Camera center: {avg_P}")

	imagetxt_list = []
	cameratxt_list = []

	db = COLMAPDatabase.connect(paths['db'])
	db.create_tables()

	print(f"Start writing to new database at '{paths['db']}'")

	cam_idx = 0

	for i, camera in tqdm(enumerate(calibration["cameras"]), desc="Reading camera calibration", total=len(calibration['cameras'])):
		# Find corresponding images
		camera_name = camera["camera_id"]
		matching_images = [f for f in filenames if camera_name in f]

		if len(matching_images) == 0:
			print(f"Missing image source for camera {camera_name}. Skipping this camera.")
			continue

		# image_name = "cam" + str(cam_idx).zfill(2) + ".jpg"
		image_name = camera_name

		img = Image.open(os.path.join(paths['input'], image_name))
		width, height = img.size

		# Extract pose information from the JSON file
		view_matrix = np.array(camera["extrinsics"]["view_matrix"], dtype=np.float64).reshape((4, 4)) # View Matrix is given in World-To-Camera Space (i think)
		#view_matrix = rotate_z_view_matrix(view_matrix)

		camera_matrix = np.array(camera["intrinsics"]["camera_matrix"], dtype=np.float64).reshape((3, 3))

		if not args.camera in ["PINHOLE", "SIMPLE_PINHOLE"]:
			distortion_coefficients = np.array(camera["intrinsics"]["distortion_coefficients"], dtype=np.float64)

		# Camera rotation and translation
		R = view_matrix[:3, :3]
		T = view_matrix[:3, 3]
		Q = rotmat2qvec(R)

		# Shift camera positions
		#P = -R.T @ T
		#P -= avg_P
		#T = -R @ P

		# focal length
		f_x = camera_matrix[0, 0]
		f_y = camera_matrix[1, 1]

		# principal point
		c_x = camera_matrix[0, 2]
		c_y = camera_matrix[1, 2]
		
		# Correct the parameters if a camera has incorrect values for some reason...
		json_width = int(camera["intrinsics"]["resolution"][0] * s)
		json_height = int(camera["intrinsics"]["resolution"][1] * s)

		if json_width != width or json_height != height:
			x_correction = float(width) / float(json_width)
			y_correction = float(height) / float(json_height)

			f_x *= x_correction
			f_y *= y_correction
			c_x *= x_correction
			c_y *= y_correction

		# Downscale
		f_x *= s
		f_y *= s

		c_x *= s
		c_y *= s

		# Write camera and image data into database
		if args.camera == "SIMPLE_PINHOLE":
			params = np.array([f_x, c_x, c_y])
		elif args.camera == "PINHOLE":
			params = np.array([f_x, f_y, c_x, c_y])
		elif args.camera == "OPENCV":
			params = np.array([f_x, f_y, c_x, c_y, distortion_coefficients[0], distortion_coefficients[1], distortion_coefficients[2], distortion_coefficients[3]])
			# params = np.array([f_x, f_y, c_x, c_y, 0, 0, 0, 0])
		elif args.camera == "RADIAL":
			params = np.array([f_x, c_x, c_y, distortion_coefficients[0], distortion_coefficients[1]])
		elif args.camera == "RADIAL_FISHEYE":
			params = np.array([f_x, c_x, c_y, distortion_coefficients[0], distortion_coefficients[1]])
			# params = np.array([f_x, c_x, c_y, 0, 0])

		camera_id = db.add_camera(camera_model, width, height, params)

		#image_id = db.add_image(image_name, camera_id, Q, T, image_id=i+1) # For COLMAP <= 3.9
		image_id = db.add_image(image_name, camera_id, image_id=i+1) # For COLMAP >= 3.10

		#db.add_pose_prior(image_id, P)

		db.commit()

		# Append lines for images.txt and cameras.txt
		Q_string = " ".join([str(q) for q in Q])
		T_string = " ".join([str(t) for t in T])
		params_string = " ".join([str(num) for num in params])

		image_line = f"{image_id} {Q_string} {T_string} {camera_id} {image_name}\n"
		imagetxt_list.append(image_line)
		imagetxt_list.append("\n")

		camera_line = f"{camera_id} {args.camera} {width} {height} {params_string}\n"
		cameratxt_list.append(camera_line)
		
		cam_idx += 1

	db.close()

	print("Done writing to database")
	
	# Write prepared data into images.txt and cameras.txt
	write_lines_to_file(imagetxt_list, paths['imagestxt'])
	write_lines_to_file(cameratxt_list, paths['camerastxt'])
	write_lines_to_file([], paths['points3Dtxt'])

	print("Done writing text output")


def main():
	parser = argparse.ArgumentParser(prog="python pre_vci.py", description="Generates a sparse/dense point cloud from a set of input images and camera poses generated by the VCI")
	parser.add_argument("--source_path", "-s", default="", type=str, required=True, help="the path where the image files are located")
	parser.add_argument("--calibration_file", "-c", default="calibration.json", type=str, help="the path to the camera calibration file (default: %(default)s)")
	parser.add_argument("--resolution", "-r", default=1, type=int, choices=[1,2,4,8], help="downscale the image by factor r (default: %(default)s)")
	parser.add_argument("--camera", default="PINHOLE", type=str, choices=["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL", "RADIAL_FISHEYE"], help="the camera model used when extracting the poses (default: %(default)s)")
	parser.add_argument("--use_mapper", action="store_true", default=False, help="use the COLMAP mapper to extract the camera poses and ignore the calibration file (default: %(default)s)")
	parser.add_argument("--replace_images", action="store_true", default=False, help="delete previously generated input images and replace them by newly generated ones (default: %(default)s)")
	parser.add_argument("--gaussian_splatting", action="store_true", default=False, help="enable output for 3d gaussian splatting (default: %(default)s)")
	parser.add_argument("--skip_dense", action="store_true", default=False, help="skip dense reconstruction (True when --gaussian_splatting is set, default: %(default)s)")
	parser.add_argument("--remove_background", action="store_true", default=False, help="use BackgroundMattingV2 to remove the background (default: %(default)s)")
	args = parser.parse_args()

	if args.gaussian_splatting:
		args.skip_dense = True

	# Load all the paths from the passed arguments
	paths = load_paths(args)
	assert os.path.exists(paths['base'])

	# Print out the paths
	print("Set directories:")
	for k, v in paths.items():
		print(f"{k:<20} -> {v}")
	print()

	# Remove old files and directories
	if os.path.exists(paths['colmap']):
		whitelist: list[str] = []
		if not args.replace_images:
			whitelist = [paths['input'], paths['masks']]

		removed = clear_directory(paths['colmap'], whitelist)
		print("Removed files and directories:")
		print("\n".join(removed))
		print()

	print(f"Preparing data from '{paths['base']}'")
	print()

	prepare_input_images_ed3dgs(paths, args)

	images, bgs = prepare_input_images_colmap(paths, args)

	if not args.use_mapper:
		extract_poses(paths, args)
		run_colmap_poses(paths, args)
	else:
		run_colmap_mapper(paths, args)

	if not args.skip_dense:
		process_ply_file(paths['output'], paths['output_downsample'])

if __name__ == "__main__":
	main()