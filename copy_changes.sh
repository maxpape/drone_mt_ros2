#!/bin/bash

rsync -a /drone_mt_ros2/px4_changes/4006_gz_px4vision /drone_mt_ros2/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/4006_gz_px4vision
rsync -a /drone_mt_ros2/px4_changes/dds_topics.yaml /drone_mt_ros2/PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml