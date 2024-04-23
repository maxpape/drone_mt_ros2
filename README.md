### Installation and execution of the code works most seamlessly with VSCode and docker.

1. Install Docker:
    https://docs.docker.com/engine/install/ubuntu/
2. Add user to docker group:
    https://docs.docker.com/engine/install/linux-postinstall/
3. Install VSCode:
    https://code.visualstudio.com/download
4. Install Dev Container extension:
    https://code.visualstudio.com/docs/devcontainers/tutorial


### Relevant files for the MPC:

1. Drone model. Specifies system model of drone for ACADOS
```
/drone_mt_ros2/ros/src/px4_ros_com/scripts/drone_model.py
```


2. Offboard control node. Topic subscription/publishing, setpoint handling, MPC execution
```
/drone_mt_ros2/ros/src/px4_ros_com/src/examples/offboard_py/offboard_control.py
```

### Prerequisites:

1. Clone the repository
2. cd into base directory:
```
cd ./drone_mt_ros2
```
3. open VSCode:
```
code .
```
4. Press 'Reopen in Container' when VSCode prompts you to do so

5. Open integrated Terminal in VSCode

6. Execute first_run.sh script
```
./first_run.sh
```

7. Close and re-open integrated Terminal

### Code execution:

1. Start Gazebo simulator:
```
cd PX4-Autopilot
make px4_sitl gz_px4vision
```

2. Open new integrated Terminal and start uXRCE-DDS (PX4-ROS 2/DDS Bridge):
```
MicroXRCEAgent udp4 -p 8888
```

3. (only on first run or code change) Build ROS code. Open new integrated terminal:
```
cd ros
colcon build --packages-select px4_ros_com
```

4. Start MPC offboard control node:
```
ros2 run px4_ros_com offboard_control.py
```

5. Publish setopints with rqt. Launch in new integrated terminal:
```
rqt
```



