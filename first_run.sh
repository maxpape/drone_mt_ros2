#!/bin/bash


sudo apt update -y
sudo apt install -y python3-pip wget nano
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y gz-garden
pip install --user -U empy==3.3.4 pyros-genmsg setuptools
pip install symforce
pip install casadi
pip install spatial-casadi
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
cd PX4-Autopilot/
DONT_RUN = 1 make px4_sitl gz_x500

cd /drone_mt_ros2
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/


cd /drone_mt_ros2/
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
pip install -e /drone_mt_ros2/acados/interfaces/acados_template


cd /drone_mt_ros2/ros
colcon build 