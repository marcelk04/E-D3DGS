import os
import PIL
import shutil
from PIL import ImageFile

def create_dir(path: str) -> bool:
	if not os.path.exists(path):
		os.makedirs(path)
		return True
	
	return False

def write_lines_to_file(lines: list[str], path: str) -> None:
	with open(path, "w") as f:
		for line in lines:
			f.write(line)

def remove_files(path: str, filenames: list[str]) -> int:
	count = 0

	for file in filenames:
		filepath = os.path.join(path, file)

		if os.path.exists(filepath):
			os.remove(filepath)

			count += 1

	return count

def clear_directory(path: str, whitelist: list[str]) -> list[str]:
	removed: list[str] = []

	for p in os.listdir(path):
		p = os.path.join(path, p)

		if p in whitelist:
			continue

		if os.path.isdir(p):
			shutil.rmtree(p)
		else:
			os.remove(p)
		
		removed.append(p)
	
	return removed

def exec_cmd(cmd: str) -> None:
	print(f"Executing '{cmd}'")

	exit_code = os.system(cmd)

	if exit_code != 0:
		exit(exit_code)

	print()

def scale_image(img: ImageFile.ImageFile, scale: float) -> ImageFile.ImageFile:
	w, h = img.size

	img.thumbnail((scale * w, scale * h))

	return img
