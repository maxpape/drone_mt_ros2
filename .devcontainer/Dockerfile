FROM osrf/ros:humble-desktop-full

# Add vscode user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# Switch from root to user
USER $USERNAME

# Add user to video and render group to allow access to webcam and graphics
RUN sudo usermod --append --groups video $USERNAME


# Update all packages
RUN sudo apt update && sudo apt upgrade -y



# Install Git
RUN sudo apt install -y git libgl1-mesa-glx libgl1-mesa-dri 


# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source /drone_mt_ros2/ros/install/local_setup.bash" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/drone_mt_ros2/acados/lib " >> ~/.bashrc
RUN echo "export ACADOS_SOURCE_DIR=/drone_mt_ros2/acados " >> ~/.bashrc
