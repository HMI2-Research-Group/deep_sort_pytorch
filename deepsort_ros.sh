#!/bin/bash
export PYTHONPATH=""
get_me_conda
source activate deepsort
cd /home/student/PycharmProjects/deep_sort_pytorch/
/home/student/anaconda3/envs/deepsort/bin/python deepsort_ros_py.py /home/student/Videos/test.mp4 --config_detection ./configs/yolov3.yaml --display