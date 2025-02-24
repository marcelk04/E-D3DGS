import torch
import numpy as np

def to_torch(np_arr):
	return torch.as_tensor(np_arr).to(torch.float32).to(torch.device("cuda"))

def to_numpy(torch_arr):
	return np.asarray(torch_arr.cpu())

def calculate_mask(img_src, bg_src, model):
	# Change image layout (h x w x 3 -> 3 x h x w)
	img_src = np.transpose(img_src, axes=(2, 0, 1))
	bg_src = np.transpose(bg_src, axes=(2, 0, 1))

	with torch.no_grad():
		# Move images to gpu
		img = to_torch(img_src)
		bg = to_torch(bg_src)

		# Calculate mask and foreground
		mask, img = model(img[None, ...], bg[None, ...])[:2]

		# Make sure background is correctly removed
		img = img * mask

		# Remove unnecessary dimensions
		img = torch.squeeze(img)
		mask = torch.squeeze(mask)

		# Transform back to numpy arrays
		img_np = to_numpy(img)
		mask_np = to_numpy(mask)

		# Change back image layout (3 x h x w -> h x w x 3)
		img_np = np.transpose(img_np, axes=(1, 2, 0))

		return img_np, mask_np

