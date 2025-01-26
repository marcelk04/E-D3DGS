import numpy as np

def lh_to_rh(matrix):
	if matrix.shape != (4, 4):
		raise ValueError("The input matrix has to have shape (4, 4)")
	
	flip_x = np.identity(4)
	flip_x[0, 0] = -1

	matrix_inv = view_matrix_inv(matrix)

	matrix_inv = matrix_inv @ flip_x

	matrix = view_matrix_inv(matrix_inv)
	
	# Entlang der yz-Ebene flippen
	# flip_x = np.eye(3)
	# flip_x[0, 0] = -1

	# R = matrix[:3, :3]
	# T = matrix[:3, 3]

	# R_inv, P = view_matrix_inv_RT(R, T)

	# R_inv = flip_x @ R_inv
	# P = flip_x @ P

	# R, T = view_matrix_inv_RT(R_inv, P)

	# #R = flip_x @ R
	# #T = flip_x @ T

	# matrix[:3, :3] = R
	# matrix[:3, 3] = T

	return matrix

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

def rotation_matrix_to_quaternion(matrix):
	trace = np.trace(matrix)

	if trace > 0:
		s = 2.0 * np.sqrt(trace + 1.0)
		q_w = 0.25 * s
		q_x = (matrix[2, 1] - matrix[1, 2]) / s
		q_y = (matrix[0, 2] - matrix[2, 0]) / s
		q_z = (matrix[1, 0] - matrix[0, 1]) / s
	elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
		s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
		q_w = (matrix[2, 1] - matrix[1, 2]) / s
		q_x = 0.25 * s
		q_y = (matrix[0, 1] + matrix[1, 0]) / s
		q_z = (matrix[0, 2] + matrix[2, 0]) / s
	elif matrix[1, 1] > matrix[2, 2]:
		s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
		q_w = (matrix[0, 2] - matrix[2, 0]) / s
		q_x = (matrix[0, 1] + matrix[1, 0]) / s
		q_y = 0.25 * s
		q_z = (matrix[1, 2] + matrix[2, 1]) / s
	else:
		s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
		q_w = (matrix[1, 0] - matrix[0, 1]) / s
		q_x = (matrix[0, 2] + matrix[2, 0]) / s
		q_y = (matrix[1, 2] + matrix[2, 1]) / s
		q_z = 0.25 * s

	return np.array([q_w, q_x, q_y, q_z])
