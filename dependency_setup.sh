#!/bin/bash
# This has to setup after every restart of the machine

## This assumed deepsort is the conda environment name
cd /tmp
mkdir catkin_ws && cd catkin_ws
mkdir src && cd src
git clone -b melodic-devel git@github.com:ros/geometry2.git
git clone -b melodic-devel git@github.com:ros/geometry.git
cd ..
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/home/student/anaconda3/envs/deepsort/bin/python \
            -DPYTHON_INCLUDE_DIR=/home/student/anaconda3/envs/deepsort/include/python3.7m \
            -DPYTHON_LIBRARY=/home/student/anaconda3/envs/deepsort/lib/libpython3.7m.so