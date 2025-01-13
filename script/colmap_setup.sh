#!/bin/bash
# install colmap for preprocess, work with python3.8
conda create -n colmapenv python=3.8
conda activate colmapenv
pip install opencv-python-headless
pip install tqdm
pip install natsort
pip install Pillow
pip install scipy
pip install scikit-image
# just some files need torch be installed.
conda install pytorch==1.12.1 -c pytorch -c conda-forge
conda config --set channel_priority false
conda install colmap=3.11.1 -c conda-forge