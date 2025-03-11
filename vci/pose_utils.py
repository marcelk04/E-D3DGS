import numpy as np

def rotate_z_view_matrix(matrix):
	R = matrix[:3, :3]
	T = matrix[:3, 3]

	# in camera to world space
	R_inv, P = view_matrix_inv_RT(R, T)

	R_z = np.array([
		[0, 1, 0],
		[-1, 0, 0],
		[0, 0, 1]
	])

	R_inv = R_z @ R_inv
	P = R_z @ P

	# in world to camera space
	R, T = view_matrix_inv_RT(R_inv, P)

	matrix[:3, :3] = R
	matrix[:3, 3] = T

	return matrix

def view_matrix_inv_RT(R, T):
	R_inv = R.T
	T_inv = -np.dot(R_inv, T)

	return (R_inv, T_inv)

def view_matrix_inv(matrix):
	R = matrix[:3, :3]
	T = matrix[:3, 3]

	R_inv, T_inv = view_matrix_inv_RT(R, T)

	matrix[:3, :3] = R_inv
	matrix[:3, 3] = T_inv

	return matrix

def normalize_poses(poses):
	cameras = poses["cameras"]

	cam_origin = [cam for cam in cameras if cam["camera_id"] == "cam02.jpg" or cam["camera_id"] == "C0004"][0]
	origin_view_matrix = np.array(cam_origin["extrinsics"]["view_matrix"]).reshape((4, 4))
	p_origin = origin_view_matrix[:3, 3]
	inv_rot_origin = np.linalg.inv(origin_view_matrix[:3, :3])

	for cam in cameras:
		view_matrix = view_matrix_inv(np.array(cam["extrinsics"]["view_matrix"]).reshape((4,4)))

		P = view_matrix[:3, 3]
		R = view_matrix[:3, :3]

		#P -= p_origin
		P = inv_rot_origin @ P

		R = inv_rot_origin @ R

		view_matrix[:3, 3] = P
		view_matrix[:3, :3] = R

		cam["extrinsics"]["view_matrix"] = view_matrix_inv(view_matrix).flatten().tolist()

	return poses
	
