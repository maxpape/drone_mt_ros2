#!/bin/bash

git submodule update --init --recursive
sudo apt update -y
sudo apt install -y python3-pip wget nano


pip install --user -U empy==3.3.4 pyros-genmsg setuptools
pip install symforce
pip install casadi
pip install spatial-casadi
pip install GPy









cd /drone_mt_ros2/ros
colcon build 