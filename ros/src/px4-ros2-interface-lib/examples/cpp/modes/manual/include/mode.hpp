/****************************************************************************
 * Copyright (c) 2023 PX4 Development Team.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************/
#pragma once

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/rates.hpp>
#include <px4_ros2/control/setpoint_types/experimental/attitude.hpp>
#include <px4_ros2/control/setpoint_types/direct_actuators.hpp>
#include <px4_ros2/control/peripheral_actuators.hpp>
#include <px4_ros2/utils/geometry.hpp>

#include <Eigen/Eigen>

#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals; // NOLINT

static const std::string kName = "My Manual Mode";

class FlightModeTest : public px4_ros2::ModeBase
{
public:
  explicit FlightModeTest(rclcpp::Node & node)
  : ModeBase(node, kName)
  {
    _manual_control_input = std::make_shared<px4_ros2::ManualControlInput>(*this);
    _rates_setpoint = std::make_shared<px4_ros2::RatesSetpointType>(*this);
    _attitude_setpoint = std::make_shared<px4_ros2::AttitudeSetpointType>(*this);
    _peripheral_actuator_controls = std::make_shared<px4_ros2::PeripheralActuatorControls>(*this);
    _motor_setpoint = std::make_shared<px4_ros2::DirectActuatorsSetpointType>(*this);
  }

  void onActivate() override {}

  void onDeactivate() override {}

  void updateSetpoint(float dt_s) override
  {
    

    // Example to control a servo by passing through RC aux1 channel to 'Peripheral Actuator Set 1'
    _peripheral_actuator_controls->set(_manual_control_input->aux1());
    Eigen::Matrix<float, 12, 1> controls =  {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    _motor_setpoint->updateMotors(controls);
  }

private:
  std::shared_ptr<px4_ros2::ManualControlInput> _manual_control_input;
  std::shared_ptr<px4_ros2::RatesSetpointType> _rates_setpoint;
  std::shared_ptr<px4_ros2::AttitudeSetpointType> _attitude_setpoint;
  std::shared_ptr<px4_ros2::PeripheralActuatorControls> _peripheral_actuator_controls;
  std::shared_ptr<px4_ros2::DirectActuatorsSetpointType>  _motor_setpoint;
  float _yaw{0.f};
};
