# Reconstructing Dynamic Scenes using Embedding-Based Deformable 3D Gaussian Splatting

This repository is a fork of [E-D3DGS](https://github.com/JeongminB/E-D3DGS), I just modified the code to be able to reconstruct our own scenes.

## Setup and Installation of Dependencies

Clone the repository using:

```bash
git clone --recurse-submodules https://github.com/marcelk04/E-D3DGS.git 
cd E-D3DGS
```

To install the environment for training, follow:

```bash
conda create -n ed3dgs python=3.9 
conda activate ed3dgs

pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization/
pip install -e submodules/simple-knn/ 

conda deactivate
```

For preprocessing using COLMAP, you will need to create a separate environment using:

```bash
conda env create -f colmapenv.yml
```

## Preprocessing

I tested my scripts with the scene "2024_12_12_dynamic3", but they should work for other scenes too.

To prepare the input images and create a point cloud, use:

```bash
cd vci
python pre_vci.py -s path/to/scene
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for pre_vci.py</span></summary>

#### --source_path / -s
Path to the dataset

#### --calibration_file / -c
Path to a calibration file containing the camera poses, ignored when COLMAP's mapper is used (default: calibration.json)

#### --camera
The camera type to be used by COLMAP (default and recommended: PINHOLE)

#### --use_mapper
Use the COLMAP mapper to estimate the camera poses

#### --replace_images
Delete previously prepared images and copy/process them again

#### --gaussian_splatting
Enable output for static 3D Gaussian Splatting

#### --skip_dense
Skip the dense reconstruction and only create a sparse reconstruction (True when --gaussian_splatting is set)

#### --remove_background
Use the given backgrounds to extract alpha masks and mask out the background

#### --rotate_images
Rotate the input images correctly (True when --use_mapper is set)

</details>

A sensible configuration for preprocessing can be executed by using:

```bash
cd vci
bash pre_vci.sh
```

## Training

The given training scripts can be used as described below. For easy use I also provided a bash script, which takes care of training, rendering, and optionally metrics:

```bash
bash train_vci.sh
```

To change the training parameters, modify arguments/vci/default.py or arguments/vci/2024_12_12_dynamic3.py.

Below, I included the old README for completeness' sake.


#  E-D3DGS : Embedding-Based Deformable 3D Gaussian Splatting (ECCV 2024)

[![arXiv](https://img.shields.io/badge/arXiv-2404.03613-006600)](https://arxiv.org/abs/2404.03613) 
[![project_page](https://img.shields.io/badge/project_page-68BC71)](https://jeongminb.github.io/e-d3dgs/)

[Jeongmin Bae](https://jeongminb.github.io/)<sup>1*</sup>, [Seoha Kim](https://seoha-kim.github.io/)<sup>1*</sup>, [Youngsik Yun](https://bbangsik13.github.io/)<sup>1</sup>, </br>
Hahyun Lee<sup>2 </sup>, Gun Bang<sup>2</sup>, [Youngjung Uh](https://github.com/yj-uh)<sup>1†</sup>

<sup>1</sup>Yonsei University &emsp; <sup>2</sup>Electronics and Telecommunications Research Institute (ETRI)
<br><sup>\*</sup> Equal Contributions &emsp; <sup>†</sup> Corresponding Author

---


Official repository for <a href="https://arxiv.org/abs/2404.03613">"Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting"</a><be>. <br>
Our approach employs per-Gaussian latent embeddings to predict deformation for each Gaussian and achieves a clearer representation of dynamic motion. <br>
We uploaded the checkpoints, configs, and rendered videos for paper results [here](https://drive.google.com/drive/folders/1PAaIp5cNYNpLjQ5JX0SVLh5Yn_K9UmJd?usp=sharing).

![Alt Text](https://github.com/JeongminB/E-D3DGS/blob/main/teaser.gif)

## Environmental Setup
Please follow the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.
```bash
git clone https://github.com/JeongminB/E-D3DGS.git
cd E-D3DGS
git submodule update --init --recursive

conda create -n ed3dgs python=3.9 
conda activate ed3dgs

# If submodules fail to be downloaded, refer to the repository of 3DGS  
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization/
pip install -e submodules/simple-knn/ 
```
We use `pytorch=1.13.1+cu116` in our environment.


## Data Preparation

**Downloading Datasets:**  
Please download datasets from their official websites : [HyperNerf](https://github.com/google/hypernerf/releases/tag/v0.1), [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) and [Technicolor](https://www.interdigital.com/data_sets/light-field-dataset) <br><br>
- Please remove 'cam13.mp4' and corresponding pose from <i>coffee_martini</i> scene in the Neural 3D Video dataset. <br>
- We split the entire <i>flame_salmon_1_split</i> scene into four 300-frame scenes.

<br>

**Extracting point clouds from COLMAP:** 
```bash
# setup COLMAP 
bash script/colmap_setup.sh
conda activate colmapenv 

# automatically extract the frames and reorginize them
python script/pre_n3v.py --videopath <dataset>/<scene>
python script/pre_technicolor.py --videopath <dataset>/<scene>
python script/pre_hypernerf.py --videopath <dataset>/<scene>

# downsample dense point clouds
python script/downsample_point.py \
<location>/<scene>/colmap/dense/workspace/fused.ply <location>/<scene>/points3D_downsample.ply
```


After running COLMAP, Neural 3D Video and Technicolor datasets are orginized as follows:
```
├── data
│   | n3v
│     ├── cook_spinach
│       ├── colmap
│       ├── images
│           ├── cam01
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│           ├── cam02
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```

## Training

To resize the training image, modify `-r 2` in the command line.
``` bash
# Train
python train.py -s $GT_PATH/$SCENE --configs arguments/$DATASET/$CONFIG.py --model_path $OUTPUT_PATH --expname $DATASET/$SCENE -r 2
``` 

## Rendering


``` bash
# Render test view only
python render.py --model_path $OUTPUT_PATH --configs arguments/$DATASET/$CONFIG.py --skip_train --skip_video

# Render train view, test view, and spiral path
python render.py --model_path $OUTPUT_PATH --configs arguments/$DATASET/$CONFIG.py
```

## Evaluation
Note: In our paper, we calculate FPS by measuring rendering time only (except for save_image, etc.).
``` bash
# Evaluate
python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG
```

## Note

* We provide scripts that collectively perform training, rendering, and evaluation. See the `train_<dataset_name>.sh`. 
* You will need to configure the dataset path according to your system.
* In the config file, make sure that the `total_num_frames` and `maxtime` are equal to the total number of training frames.

## Acknowledgements

This code is based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [4DGaussians](https://github.com/hustvl/4DGaussians) and [STG](https://github.com/oppo-us-research/SpacetimeGaussians). In particular, we used [4DGaussians](https://github.com/hustvl/4DGaussians) as a starting point for our study. We would like to thank the authors of these papers for their hard work. 😊

## BibTex
```
@inproceedings{bae2024ed3dgs,
    title={Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting}, 
    author={Bae, Jeongmin and Kim, Seoha and Yun, Youngsik and Lee, Hahyun and Bang, Gun and Uh, Youngjung}, 
    booktitle = {European Conference on Computer Vision (ECCV)},
    year={2024}
}
```
