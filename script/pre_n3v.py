# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os 
import cv2 
import glob 
from tqdm import tqdm
import numpy as np 
import shutil
import pickle
import sys 
import argparse
sys.path.append(".")
from script.thirdparty.my_utils import posetow2c_matrcs, rotmat2qvec
from script.thirdparty.pre_colmap import * 
from script.thirdparty.helper3dg import getcolmapsinglen3d




def extractframes(videopath: str) -> bool:
    cam = cv2.VideoCapture(videopath)
    success = True
    ctr = 0

    # TODO: doubt this does anything, path does not seem to be correct
    for i in range(300):
        if os.path.exists(os.path.join(videopath.replace(".mp4", ""), str(i) + ".png")):
            ctr += 1

    # TODO: also, wth is this condition, this is just waiting to cause an error
    if ctr == 300 or ctr == 150: # 150 for 04_truck 
        print("already extracted all the frames, skip extracting")
        return
    
    base_dir = os.path.dirname(videopath) # directory in which the videos are located
    cam_dir = os.path.basename(videopath)[:-4] # filename without file extension, e.g. .mp4
    save_dir = os.path.join(base_dir, "images", cam_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    ctr = 0
    while ctr < 300:
        try:
            _, frame = cam.read()

            filename = str(ctr).zfill(4) + ".png"
            savepath = os.path.join(save_dir, filename)

            cv2.imwrite(savepath, frame)

            ctr += 1 
        except KeyboardInterrupt:
            success = False
            print(f"extracted {ctr} frames from {videopath}")
            break
        except:
            success = False
            print(f"error while extracting frame {ctr} from '{videopath}'")

    cam.release()

    return success


def preparecolmapdynerf(folder: str) -> None:
    """
    Copies the first image of each camera into .../colmap/input/
    """

    folderlist = glob.glob(folder + "images/cam**/")

    savedir = os.path.join(folder, "colmap", "input")
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for camfolder in folderlist:
        imagepath = os.path.join(camfolder, "0000.png")
        imagesavepath = os.path.join(savedir, camfolder.split("/")[-2] + ".png") # e.g. .../colmap/input/cam00.png

        shutil.copy(imagepath, imagesavepath)


    
def convertdynerftocolmapdb(path: str) -> None:
    # see https://github.com/bmild/nerf#generating-poses-for-your-own-scenes for poses
    # also https://github.com/fyusion/llff?tab=readme-ov-file#using-your-own-poses-without-running-colmap
    originnumpy = os.path.join(path, "poses_bounds.npy")
    video_paths = sorted(glob.glob(os.path.join(path, 'cam*.mp4')))
    projectfolder = os.path.join(path, "colmap")
    #sparsefolder = os.path.join(projectfolder, "sparse/0")
    manualfolder = os.path.join(projectfolder, "manual")

    # if not os.path.exists(sparsefolder):
    #     os.makedirs(sparsefolder)
    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")

    imagetxtlist = []
    cameratxtlist = []


    dbpath = os.path.join(projectfolder, "input.db")

    if os.path.exists(dbpath):
        os.remove(dbpath)

    db = COLMAPDatabase.connect(dbpath)

    db.create_tables()


    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file) # shape: (20, 17)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # shape (20, 3, 5)

        llffposes = poses.copy().transpose(1,2,0) # shape: (3, 5, 20)

        # TODO: wieso wieder world-to-camera, posen sind in camera-to-world gegeben
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert (type(w2c_matriclist) == list)


        for i in range(len(poses)):
            cameraname = os.path.basename(video_paths[i])[:-4]#"cam" + str(i).zfill(2)
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]
            
            H, W, focal = poses[i, :, -1]
            
            colmapQ = rotmat2qvec(colmapR)
            # colmapRcheck = qvec2rotmat(colmapQ)

            imageid = str(i+1)
            cameraid = imageid
            pngname = cameraname + ".png"
            
            line =  imageid + " "

            for j in range(4):
                line += str(colmapQ[j]) + " "
            for j in range(3):
                line += str(T[j]) + " "
            line = line  + cameraid + " " + pngname + "\n"
            empltyline = "\n"
            imagetxtlist.append(line)
            imagetxtlist.append(empltyline)

            focolength = focal
            model, width, height, params = i, W, H, np.array((focolength,  focolength, W//2, H//2,))

            camera_id = db.add_camera(1, width, height, params)
            cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"
            cameratxtlist.append(cameraline)
            
            image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)
            db.commit()
        db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 





if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=300, type=int)

    args = parser.parse_args()

    videopath = args.videopath

    # TODO: both unused? ...why
    startframe = args.startframe
    endframe = args.endframe


    if startframe >= endframe:
        print("start frame must be smaller than end frame")
        quit()

    if startframe < 0 or endframe > 300:
        print("start/end frame must be in range 0-300")
        quit()

    if not os.path.exists(videopath):
        print("path does not exist")
        quit()
    
    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    
    
    #### step 1: extract frames from videos
    images_path = os.path.join(videopath, "images")

    if not os.path.exists(images_path):
        print("start extracting 300 frames from videos")

        os.makedirs(images_path)
        videoslist = sorted(glob.glob(videopath + "*.mp4"))

        for v in tqdm(videoslist):
            extractframes(v)
    else:
        print(f"{images_path} already exists, skipping frame extraction")
    
    #### step 2: prepare colmap input 
    print("start preparing colmap image input")
    preparecolmapdynerf(videopath)


    #### step 3: prepare colmap db file 
    print("start preparing colmap database input")
    convertdynerftocolmapdb(videopath)


    #### step 4: run colmap, per frame, if error, reinstall opencv-headless 
    print("start executing colmap")
    getcolmapsinglen3d(videopath)




