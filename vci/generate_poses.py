import argparse
import os
import numpy as np
import json

# dont question this ... :/
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from script.thirdparty.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from utils.pose_utils import qvec2rotmat
from vci.pose_utils import normalize_poses

def read_poses(path: str):
	extrinsics = read_extrinsics_binary(os.path.join(path, "images.bin"))
	intrinsics = read_intrinsics_binary(os.path.join(path, "cameras.bin"))

	print(f"Reading {len(extrinsics)} cameras, {len(intrinsics)} images...")

	cameras = []

	for key in extrinsics:
		extr = extrinsics[key]
		intr = intrinsics[extr.camera_id]

		view_matrix = np.identity(4)
		view_matrix[:3, :3] = qvec2rotmat(extr.qvec)
		view_matrix[:3, 3] = extr.tvec
		
		extr_obj = {}
		extr_obj["view_matrix"] = view_matrix.flatten().tolist()

		cam_matrix = np.identity(3)
		cam_matrix[0, 0] = intr.params[0] # f_x
		cam_matrix[1, 1] = intr.params[1] # f_y
		cam_matrix[0, 2] = intr.params[2] # c_x
		cam_matrix[1, 2] = intr.params[3] # c_y

		intr_obj = {}
		intr_obj["camera_matrix"] = cam_matrix.flatten().tolist()
		intr_obj["resolution"] = [intr.width, intr.height]

		cam = {}
		cam["camera_id"] = extr.name
		cam["extrinsics"] = extr_obj
		cam["intrinsics"] = intr_obj

		cameras.append(cam)

	cameras = sorted(cameras, key=lambda cam: cam["camera_id"])

	poses = {}
	poses["cameras"] = cameras

	return poses

def save_poses(poses, output: str) -> None:
	json_object = json.dumps(poses, indent=2)

	with open(output, "w") as outfile:
		outfile.write(json_object)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", "-i", type=str, required=True)
	parser.add_argument("--output", "-o", type=str, required=True)
	args = parser.parse_args()

	poses = read_poses(args.input)
	poses = normalize_poses(poses)
	save_poses(poses, args.output)

	print("Done.")

if __name__ == "__main__":
	main()