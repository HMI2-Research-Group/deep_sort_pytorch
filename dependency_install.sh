#!/bin/bash
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y -c anaconda scipy
conda install -y -c anaconda cython
conda install -y -c conda-forge scikit-learn
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge ros-rospy
conda install -y -c conda-forge ros-geometry-msgs
conda install -y -c anaconda pillow
conda install -y -c conda-forge easydict
conda install -y -c conda-forge pyyaml
conda install -y -c conda-forge tabulate
conda install -y -c conda-forge yacs
conda install -y -c conda-forge ros-sensor-msgs
conda install -y -c conda-forge ros-nav-msgs
conda install -y -c conda-forge ros-cv-bridge
pip install opencv-python
pip install opencv-contrib-python
pip install Vizer
pip install faiss-cpu
pip install tensorboard
pip install gdown
pip install termcolor
pip install openmim
mim install mmdet

# The HASH of the last commit: c791220cefd0abf02c6719e2ce0fea465857a88e
git clone git@github.com:ros-perception/vision_opencv.git
cd vision_opencv
cd image_geometry
python setup.py install



