#!/bin/bash
# This has to setup after every restart of the machine

## This assumed deepsort is the conda environment name
cd /tmp
mkdir catkin_ws && cd catkin_ws
mkdir src && cd src
git clone -b melodic-devel git@github.com:ros/geometry2.git
git clone -b melodic-devel git@github.com:ros/geometry.git
cd ..
# Basically we will be using everything from ROS, except the Python interpreter which we will be using
# from anaconda. Then in our code we will be adding the dist-packages to the PYTHONPATH
# Example: sys.path.append("/tmp/catkin_ws/devel/lib/python3/dist-packages")
# Pro Tip: If you do this in a normal folder rather than temp folder, even Pylance can recognize the Python packages
# and it will be able to import them.
source /opt/ros/melodic/setup.bash
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=$HOME/anaconda3/envs/deepsort/bin/python \
            -DPYTHON_INCLUDE_DIR=$HOME/anaconda3/envs/deepsort/include/python3.7m \
            -DPYTHON_LIBRARY=$HOME/anaconda3/envs/deepsort/lib/libpython3.7m.so