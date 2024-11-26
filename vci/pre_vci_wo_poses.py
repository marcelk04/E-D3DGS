import argparse
import os

def exec_cmd(cmd):
	print(f"Executing '{cmd}'")

	exit_code = os.system(cmd)

	if exit_code != 0:
		exit(exit_code)

	print()

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--workspace", default="", type=str)
	parser.add_argument("--imagepath", default="", type=str)

	args = parser.parse_args()

	workspace = args.workspace
	image_path = args.imagepath

	if (image_path == ""):
		image_path = os.path.join(workspace, "images")

	print("Workspace: ", workspace)
	print("Image path: ", image_path)

	colmap_path = os.path.join(workspace, "colmap")
	if not os.path.exists(colmap_path):
		os.makedirs(colmap_path)

	distorted_path = os.path.join(colmap_path, "distorted", "sparse")
	if not os.path.exists(distorted_path):
		os.makedirs(distorted_path)

	dense_path = os.path.join(colmap_path, "dense", "workspace")
	if not os.path.exists(dense_path):
		os.makedirs(dense_path)

	fused_path = os.path.join(dense_path, "fused.ply")

	db_path = os.path.join(colmap_path, "database.db")

	# Delete db and create a new one
	if os.path.exists(db_path):
		print(f"Overwriting '{db_path}'")

		os.remove(db_path)

		db = open(db_path, "w")
		db.close()

	# Colmap
	feature_extract = f"colmap feature_extractor --database_path {db_path} --image_path {image_path}"
	exec_cmd(feature_extract)

	feature_matcher = f"colmap exhaustive_matcher --database_path {db_path} --TwoViewGeometry.min_num_inliers=10"
	#exec_cmd(feature_matcher)

	mapper = f"colmap mapper --database_path {db_path} --image_path {image_path} --output_path {distorted_path} --Mapper.min_num_matches=10 --Mapper.init_min_num_inliers=15"
	#exec_cmd(mapper)

	tri_and_map = f"colmap point_triangulator --database_path {db_path} --image_path {image_path} --input_path {distorted_path}/0 --output_path {distorted_path} --Mapper.ba_global_function_tolerance=0.000001"
	#exec_cmd(tri_and_map)

	image_undistortion = f"colmap image_undistorter --image_path {image_path} --input_path {distorted_path}/0 --output_path {dense_path} --output_type COLMAP"
	#exec_cmd(image_undistortion)

	patch_match_stereo = f"colmap patch_match_stereo --workspace_path {dense_path}"
	#exec_cmd(patch_match_stereo)

	stereo_fusion = f"colmap stereo_fusion --workspace_path {dense_path} --output_path {fused_path}"
	#exec_cmd(stereo_fusion)

	print("All done! Output is in ", fused_path)

if __name__ == "__main__":
	main()