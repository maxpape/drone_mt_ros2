#!/usr/bin/env python3

import rclpy
import numpy as np
import scipy.linalg
import scipy.interpolate
from casadi import vertcat
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, ActuatorMotors, VehicleOdometry
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from drone_model import export_drone_ode_model

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_nmpc')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.motor_command_publisher = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)



        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.asarray([1,0,0,0])
        self.angular_velocity = np.zeros(3)
        self.current_state = np.asarray([0,0,0,1,0,0,0,0,0,0,0,0,0])
        self.ocp_solver = None
        self.position_setpoint = np.zeros(3)

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        
        
        # Declare a parameter with a descriptor for dynamic reconfiguration
        motor_speed_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,  # Specify the type as double
            description='Speed of the motor',     # Description of the parameter
            additional_constraints='Range: -1.0 to 1.0',
            floating_point_range=[FloatingPointRange(
                from_value=-1.0,  # Minimum value
                to_value=1.0,     # Maximum value
                step=0.01         # Step size (optional)
            )]# Constraints (optional)
        )
        
        self.declare_parameters(
        namespace='',
        parameters=[
            ('motor_speed_0', 0.0, motor_speed_descriptor),
            ('motor_speed_1', 0.0, motor_speed_descriptor),
            ('motor_speed_2', 0.0, motor_speed_descriptor),
            ('motor_speed_3', 0.0, motor_speed_descriptor)
        ]
        )
        
        
        # Declare a parameter with a descriptor for dynamic reconfiguration
        position_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,  # Specify the type as double
            description='Desired position',     # Description of the parameter
            additional_constraints='Range: -50 to 50',
            floating_point_range=[FloatingPointRange(
                from_value=-50.0,  # Minimum value
                to_value=50.0,     # Maximum value
                step=0.1         # Step size (optional)
            )]# Constraints (optional)
        )
        
        self.declare_parameters(
        namespace='',
        parameters=[
            ('position_x', 0.0, position_descriptor),
            ('position_y', 0.0, position_descriptor),
            ('position_z', 0.0, position_descriptor),
        ]
        )
        
        
        print("test")
        
        self.speed = np.zeros(4)
        self.N_horizon = 40
        self.Tf = 2
        self.nx = 13
        self.nu = 4
        self.Tmax = 10
        self.Tmin = 0
        
        
        self.m = 1.5
        self.g = 9.81
        self.jxx = 0.029125
        self.jyy = 0.029125
        self.jzz = 0.055225
        self.d_x0 = 0.107
        self.d_x1 = 0.107 
        self.d_x2 = 0.107 
        self.d_x3 = 0.107 
        self.d_y0 = 0.0935
        self.d_y1 = 0.0935
        self.d_y2 = 0.0935
        self.d_y3 = 0.0935
        self.c_tau = 0.000806428
        
        
        self.parameters = np.asarray([self.m,
                                        self.g,
                                        self.jxx,
                                        self.jyy,
                                        self.jzz,
                                        self.d_x0,
                                        self.d_x1,
                                        self.d_x2, 
                                        self.d_x3, 
                                        self.d_y0,
                                        self.d_y1,
                                        self.d_y2,
                                        self.d_y3,
                                        self.c_tau])
        
        
        
    def setup_mpc(self):
        ocp = AcadosOcp()
        
        
        # set model
        model = export_drone_ode_model()
        ocp.model = model

        
        
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        ocp.dims.N = self.N_horizon
        
        ocp.parameter_values = self.parameters
    
        # set cost module
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        
        Q_mat = np.zeros((13,13))
        Q_mat[0,0] = 2
        Q_mat[1,1] = 2
        Q_mat[2,2] = 3
        R_mat = np.eye(4)
        
        Q_mat_final = np.eye(13)
        Q_mat_final[0,0] = 2
        Q_mat_final[1,1] = 2
        Q_mat_final[2,2] = 3        

        
        
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.W_e = Q_mat_final

        ocp.model.cost_y_expr = vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x

        yref = np.zeros((ny, ))
        yref[3] = 1 
        ocp.cost.yref  = yref

        yref_e = np.zeros((ny_e, ))
        yref_e[3] = 1
        ocp.cost.yref_e = yref_e
        
        Tmin = self.Tmin
        Tmax = self.Tmax
        
        # set constraints
        ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
        ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
    
                   
        ocp.constraints.x0 = self.current_state
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        
        ocp.solver_options.qp_solver_cond_N = self.N_horizon

        # set prediction horizon
        ocp.solver_options.tf = self.Tf

        solver_json = 'acados_ocp_' + model.name + '.json'
        self.ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)
           
        

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
        
    def vehicle_odometry_callback(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber."""
        
        self.position = self.NED_to_ENU(vehicle_odometry.position)
        self.velocity = self.NED_to_ENU(vehicle_odometry.velocity)
        self.attitude = self.NED_to_ENU(vehicle_odometry.q)
        self.angular_velocity = self.NED_to_ENU(vehicle_odometry.angular_velocity)
        
        self.current_state[0:3] = self.position
        self.current_state[3:7] = self.attitude
        self.current_state[7:10] = self.velocity
        self.current_state[10:13] = self.angular_velocity
    
    def NED_to_ENU(self, input_array):
        """
        Rotates a 3D vector or a quaternion by 180 degrees around the x-axis.

        Parameters:
        input_array (np.ndarray): The input to be rotated. Should be a NumPy array of shape (3,) for a 3D vector or (4,) for a quaternion.

        Returns:
        np.ndarray: The rotated 3D vector or quaternion.
        """
        if input_array.shape == (3,):
            # Handle as a 3D vector
            # A 180-degree rotation around the x-axis flips the signs of the y and z components
            rotated_array = input_array * np.array([1, -1, -1])
        elif input_array.shape == (4,):
            # Handle as a quaternion
            # For a 180-degree rotation around the x-axis, the quaternion is [0, 1, 0, 0]
            # This effectively flips the signs of the y and z components of the vector part
            rotated_array = input_array * np.array([1, -1, -1, -1])
        else:
            raise ValueError("Input array must be either a 3D vector or a quaternion (shape (3,) or (4,)).")
        
        return rotated_array
    
    
    def set_mpc_target_pos(self):
    
            
        yref = np.zeros((self.nx+self.nu, ))
        yref[0:3] = self.position_setpoint
        yref[3] = 1
        
        yref_e = np.zeros((self.nx, ))
        yref_e[0:3] = self.position_setpoint
        yref[3] = 1
        
        for j in range(self.N_horizon):
            
            self.ocp_solver.set(j, "yref", yref)
            self.ocp_solver.set(j, "p", self.parameters)
        self.ocp_solver.set(self.N_horizon, "yref", yref_e)
        self.ocp_solver.set(self.N_horizon, "p", self.parameters)
        
    def map_logarithmic(self, input_value):
        # Ensure the input is within the expected range
        if input_value < 0 or input_value > 10:
            raise ValueError("Input value must be between 0 and 10")
        
        # Normalize input from 0-10 to 1-11
        normalized_input = input_value + 1
        
        # Apply logarithm with base 2
        log_output = np.log2(normalized_input)
        
        # Scale the output from 0 to slightly above 3 to 0-1
        scaled_output = log_output / np.log2(11)
        
        return scaled_output

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.direct_actuator = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    

    def publish_motor_command(self, control):
        """Publish the motor command setpoint."""
        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        msg.control[0] = control[0]  # Motor 1
        msg.control[1] = control[1]  # Motor 2
        msg.control[2] = control[2]  # Motor 3
        msg.control[3] = control[3]  # Motor 4
        
        self.motor_command_publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % control)
    
    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    
    def timer_callback(self):
        self.publish_offboard_control_heartbeat_signal()
        
        
        
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
        elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            params = self.get_parameters(
            ['position_x', 'position_y', 'position_z'])
            self.position_setpoint = np.asarray([p.value for p in params])
            self.set_mpc_target_pos()
            
            U = self.ocp_solver.solve_for_x0(x0_bar = self.current_state)
            
            command = np.asarray([self.map_logarithmic(u) for u in U])
            
            #params = self.get_parameters(
            #['motor_speed_0', 'motor_speed_1', 'motor_speed_2', 'motor_speed_3'])
            #self.speed = np.asarray([p.value for p in params])
            self.publish_motor_command(command)
            
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
        
        

def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    offboard_control.setup_mpc()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
