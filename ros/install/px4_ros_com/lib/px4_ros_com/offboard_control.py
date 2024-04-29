#!/usr/bin/env python3

import rclpy
from rcl_interfaces.msg import SetParametersResult
import numpy as np
import scipy.linalg
import scipy.interpolate
import time
import collections
from casadi import SX, vertcat, Function, sqrt, norm_2, dot, cross, atan2, if_else
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleStatus, ActuatorMotors, VehicleOdometry, ActuatorOutputs, SensorCombined
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange
from acados_template import AcadosOcp, AcadosOcpSolver
from drone_model import export_drone_ode_model


np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def euler_to_quaternion_numpy(rpy):
    """
    Convert Euler angles to quaternion.

    Parameters:
    rpy : np.ndarray roll, pitch, yaw

    Returns:
    np.ndarray
        Quaternion [w, x, y, z] representing the rotation.
    """
    roll, pitch, yaw = rpy
    # Create a rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    
    # Convert the rotation object to quaternion (scalar-last format)
    q = r.as_quat()
    
    
    return np.array([q[3], q[0], q[1], q[2]] ) / np.linalg.norm(q)

def quaternion_to_euler_numpy(q):
    """Convert quaternion to euler angles

    Args:
        q (np.ndarray): Input quaternion 

    Returns:
        no.ndarray: Array containg orientation in euler angles [roll, pitch, yaw]
    """
    quat = np.zeros(4)
    quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]

    rotation = R.from_quat(quat)

    return rotation.as_euler("xyz", degrees=True)

def quaternion_to_euler_casadi(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x-axis, pitch is rotation around y-axis,
        and yaw is rotation around z-axis.
        """
        quat = SX.sym("quat", 4)
        quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]
        rotation = sc.Rotation.from_quat(quat)
        
        
        return rotation.as_euler('xyz')



def quaternion_inverse_numpy(q):
    """Invert a quaternion given as a numpy expression

    Args:
        q (np.ndarray): input quaternion

    Returns:
        np.ndarray: inverted quaternion
    """

    return np.array([1, -1, -1, -1]) * q / (np.linalg.norm(q) ** 2)

def quaternion_product_numpy(q, p):
    """Multiply two quaternions given as numpy arrays

    Args:
        q (np.ndarray): input quaternion q
        p (np.ndarray): input quaternion p

    Returns:
        np.ndarray: output quaternion
    """
    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return np.concatenate((s, v), axis=None)

def quat_rotation_numpy(v, q):
    """Rotates a vector v by the quaternion q

    Args:
        v (np.ndarray): input vector
        q (np.ndarray): input quaternion

    Returns:
        np.ndarray: rotated vector
    """

    p = np.concatenate((0.0, v), axis=None)
    p_rotated = quaternion_product_numpy(
        quaternion_product_numpy(q, p), quaternion_inverse_numpy(q)
    )
    return p_rotated[1:4]


def multiply_quaternions_casadi(q, p):
    """Multiply two quaternions given as casadi expressions

    Args:
        q (casadi SX): quaternion 1
        p (casadi SX): quaternion 2

    Returns:
        casadi SX: resulting quaternion
    """
    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - dot(v1, v2)
    v = s1 * v2 + s2 * v1 + cross(v1, v2)
    return vertcat(s, v)

def quaternion_inverse_casadi(q):
    """Invert a quaternion given as a casadi expression

    Args:
        q (casadi SX): input quaternion

    Returns:
        casadi SX: inverted quaternion
    """

    return SX([1, -1, -1, -1]) * q / (norm_2(q)**2)

def quaternion_error(q_ref, q):
    """Calculate the quaternion error between a reference quaternion q_ref and an origin quaternion q

    Args:
        q_ref (casadi SX): reference quaternion
        q (casadi SX): origin quaternion

    Returns:
        casadi SX: elements x, y, and z from error quaternion (w neglected, since norm(unit quaternion)=1; not suitable for error calculation)
    """
    q_error = multiply_quaternions_casadi(q_ref, quaternion_inverse_casadi(q))

    if_else(q_error[0] >= 0, SX([1, -1, -1, -1])*q_error, SX([1, 1, 1, 1])*q_error, True)
    

    return q_error[1:4]

    

def error_function(x, y_ref):
    """Error function for MPC
        difference of position and attitude from reference
        use sub-function for calculating quaternion error

    Args:
        x (casadi SX): current state of position and attitude
        y_ref (casadi SX): desired reference

    Returns:
        casadi SX: vector containing position and attiude error (attitude error only regarding yaw)
    """
    p_ref = y_ref[0:3]
    q_ref = y_ref[3:7]
    
    
    p_err = x[0:3] - p_ref
    q_err = quaternion_error(x[3:7], q_ref)[2]
    
    
    return vertcat(p_err, q_err)



class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode with MPC."""

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
        self.motor_command_publisher_pseudo = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors_pseudo', qos_profile)

        # Create subscribers
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_motor_subscriber = self.create_subscription(
            ActuatorOutputs, '/fmu/out/actuator_outputs', self.vehicle_motor_callback, qos_profile)
        self.vehicle_imu_subscriber = self.create_subscription(
            SensorCombined, '/fmu/out/sensor_combined', self.vehicle_imu_callback, qos_profile)



        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_status = VehicleStatus()
        self.ocp_solver = None
        
        # variables for ACADOS MPC
        self.N_horizon = 40
        self.Tf = 2
        self.nx = 17
        self.nu = 4
        self.Tmax = 7
        self.Tmin = 1
        self.vmax = 3
        self.angular_vmax = 1.5
        self.max_angle_q = 0.2
        self.max_motor_rpm = 1100
        
        # parameters for system model
        self.m = 1.5
        
        self.g = -9.81
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
        self.hover_thrust = -self.g*self.m/4
        
        self.params = np.asarray([self.m,
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
        
        
        
        
        
        
        #state variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.asarray([np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2])
        self.angular_velocity = np.zeros(3)
        self.thrust = np.ones(4)*self.hover_thrust
        self.current_state = np.concatenate((self.position, self.attitude, self.velocity, self.angular_velocity, self.thrust, self.get_clock().now().nanoseconds), axis=None)
        self.state_history = collections.deque(maxlen=20)
        self.update_current_state()
        
        
        # imu data
        self.linear_accel = np.zeros(3)
        self.angular_accel = np.zeros(3)
        self.imu_data = np.concatenate((self.linear_accel, self.angular_accel, self.get_clock().now().nanoseconds), axis=None)
        
        
        #setpoint variables
        self.position_setpoint = np.array([0,0,0])
        self.velocity_setpoint = np.zeros(3)
        self.attitude_setpoint = np.asarray([np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2])
        self.roll_setpoint = 0
        self.pitch_setpoint = 0
        self.yaw_setpoint = 0
        self.angular_velocity_setpoint = np.zeros(3)
        self.setpoint = np.concatenate((self.position_setpoint, self.attitude_setpoint, self.velocity_setpoint, self.angular_velocity_setpoint), axis=None)
        self.update_setpoint()
        
        
        
        self.parameters = np.concatenate((self.params, self.setpoint), axis=None)

        # Create a timer to publish control commands
        self.timer = self.create_timer(self.Tf/self.N_horizon, self.timer_callback)
        
        
        
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
        
        
        # Declare a parameter with a descriptor for dynamic reconfiguration
        angle_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,  # Specify the type as double
            description='Desired pangle',     # Description of the parameter
            additional_constraints='Range: -180 to 180',
            floating_point_range=[FloatingPointRange(
                from_value=-100.0,  # Minimum value
                to_value=100.0,     # Maximum value
                step=0.1         # Step size (optional)
            )]# Constraints (optional)
        )
        
        self.declare_parameters(
        namespace='',
        parameters=[
            ('roll', 0.0, angle_descriptor),
            ('pitch', 0.0, angle_descriptor),
            ('yaw', -90.0, angle_descriptor),
        ]
        )
        
        self.add_on_set_parameters_callback(self.parameter_callback)
        
    def update_current_state(self):
        """aggregates individual states to combined state of system
        """
        
        self.current_state = np.concatenate((self.position, self.attitude, self.velocity, self.angular_velocity, self.thrust, self.get_clock().now().nanoseconds), axis=None)
        self.state_history.appendleft(self.current_state) 
        
    def update_setpoint(self, fields="", p=np.array([0,0,0]), q=np.array([1,0,0,0]), v=np.array([0,0,0]), w=np.array([0,0,0]), roll=0.0, pitch=0.0, yaw=0.0):
        """Updates one ore more reference setpoint values

        Args:
            fields (str, optional): what references shall be updated. Defaults to "".
            p (np.ndarray, optional): only set if specified in "fields". reference for position. Defaults to np.array([0,0,0]).
            q (np.ndarray, optional): only set if specified in "fields". reference for attitude. Defaults to np.array([1,0,0,0]).
            v (np.ndarray, optional): only set if specified in "fields". reference for velocity. Defaults to np.array([0,0,0]).
            w (np.ndarray, optional): only set if specified in "fields". reference for angular velocity. Defaults to np.array([0,0,0]).
            roll  (float, optional):  only set if specified in "fields". reference for roll. Defaults to 0
            pitch (float, optional):  only set if specified in "fields". reference for pitch. Defaults to 0
            yaw   (float, optional):  only set if specified in "fields". reference for yaw. Defaults to 0
        """
        if "p" in fields:
            self.position_setpoint = p
        if "q" in fields:
            self.attitude_setpoint = q
        if "v" in fields:
            self.velocity_setpoint = v
        if "w" in fields:
            self.angular_velocity_setpoint = w
        if "roll" in fields:
            self.roll_setpoint = roll
            self.attitude_setpoint = euler_to_quaternion_numpy(np.array([self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint]))
        if "pitch" in fields:
            self.pitch_setpoint = pitch
            self.attitude_setpoint = euler_to_quaternion_numpy(np.array([self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint]))
        if "yaw" in fields:
            self.yaw_setpoint = yaw
            self.attitude_setpoint = euler_to_quaternion_numpy(np.array([self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint]))
        
        self.setpoint = np.concatenate((self.position_setpoint, self.attitude_setpoint, self.velocity_setpoint, self.angular_velocity_setpoint), axis=None)
    
    def set_mpc_target_pos(self):
             
        parameters = np.concatenate((self.params, self.setpoint), axis=None)
        for j in range(self.N_horizon):
            self.ocp_solver.set(j, "p", parameters)
        
        self.ocp_solver.set(self.N_horizon, "p", parameters)  
    
    
    def parameter_callback(self, params):
        
        for param in params:
            if param.name == "position_x":
                self.position_setpoint[0] = param.value
            elif param.name == "position_y":
                self.position_setpoint[1] = param.value
            elif param.name == "position_z":
                self.position_setpoint[2] = param.value
            elif param.name == "roll":
                self.roll_setpoint = param.value
            elif param.name == "pitch":
                self.pitch_setpoint = param.value
            elif param.name == "yaw":
                self.yaw_setpoint = param.value
         
               
        self.attitude_setpoint = euler_to_quaternion_numpy(np.array([self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint]))        
        
        self.update_setpoint()
        print('New target setpoint: {}'.format(self.setpoint))
        
        self.set_mpc_target_pos()
        
        return SetParametersResult(successful=True)

    
    
       
    def setup_mpc(self):
        
        ocp = AcadosOcp()
        
        # set model
        model = export_drone_ode_model()
        ocp.model = model
        

        ocp.dims.N = self.N_horizon
        ocp.parameter_values = self.parameters
        
        
        # define weighing matrices
        Q_p= np.diag([10,10,100])
        Q_q= np.eye(1)*100
        Q_mat = scipy.linalg.block_diag(Q_p, Q_q)
    
        R_U = np.eye(4)
        
        Q_p_final = np.diag([10,10,100])
        Q_q_final = np.eye(1)*100
        Q_mat_final = scipy.linalg.block_diag(Q_p_final, Q_q_final)
        
        
        
        # set cost module
        x = ocp.model.x[0:7]
        u = ocp.model.u
        
        ref = ocp.model.p[14:21]
        
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        
        ocp.model.cost_expr_ext_cost = error_function(x, ref).T @ Q_mat @ error_function(x,ref) + u.T @ R_U @ u 
        ocp.model.cost_expr_ext_cost_e = error_function(x, ref).T @ Q_mat_final @ error_function(x, ref)
        
        
        # set constraints
        Tmin = self.Tmin
        Tmax = self.Tmax
        vmax = self.vmax
        angular_vmax = self.angular_vmax
        
        # input constraints        
        ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
        ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
         
        # state constraints     
        ocp.constraints.lbx = np.array([-self.max_angle_q, -self.max_angle_q, -vmax, -vmax, -vmax, -angular_vmax, -angular_vmax, -angular_vmax])
        ocp.constraints.ubx = np.array([+self.max_angle_q, +self.max_angle_q, +vmax, +vmax, +vmax, +angular_vmax, +angular_vmax, +angular_vmax])
        ocp.constraints.idxbx = np.array([4, 5, 7, 8, 9, 10, 11, 12])
        
        
        # set initial state
    
        ocp.constraints.x0 = self.current_state[:-1]
                
        

        
        
        # set prediction horizon
        ocp.solver_options.qp_solver_cond_N = self.N_horizon
        ocp.solver_options.tf = self.Tf
        
        # set solver options
        ocp.solver_options.levenberg_marquardt = 15.0
        ocp.solver_options.qp_solver_warm_start = 2
        
        
        # create ACADOS solver
        solver_json = 'acados_ocp_' + model.name + '.json'
        self.ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)
        
        
        
        
        



    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
        
    def vehicle_odometry_callback(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber.
        Updates position, velocity, attitude and angular velocity

        Args:
            vehicle_odometry (px4_msgs.msg.VehicleOdometry): odometry message from drone
        """
        
        
        self.position = self.NED_to_ENU(vehicle_odometry.position)
        self.velocity = self.NED_to_ENU(vehicle_odometry.velocity)
        self.attitude = self.NED_to_ENU(vehicle_odometry.q)
        self.angular_velocity = self.NED_to_ENU(vehicle_odometry.angular_velocity)
        
        
        self.update_current_state()
              
    
    
    def vehicle_motor_callback(self, motor_output):
        """Callback function for motor rpm

        Args:
            motor_output (px4_msgs.msg.ActuatorMotors): motor rpm from drone
        """
        
        m0 = motor_output.output[0]
        m1 = motor_output.output[1]
        m2 = motor_output.output[2]
        m3 = motor_output.output[3]
        
        thrust = np.zeros(4)
        
        
        thrust[0] = self.inverse_map_thrust(m0)
        thrust[1] = self.inverse_map_thrust(m1)
        thrust[2] = self.inverse_map_thrust(m2)
        thrust[3] = self.inverse_map_thrust(m3)
        
        
        self.thrust = thrust
        self.update_current_state()
        
    def vehicle_imu_callback(self, imu_data):
        
        linear_accel_body = self.NED_to_ENU(imu_data.accelerometer_m_s2)
        linear_accel_body[0] = -linear_accel_body[0]
        linear_accel_body[1] = -linear_accel_body[1]
        linear_accel_body[2] -= 9.81
        angular_accel_body = self.NED_to_ENU(imu_data.gyro_rad)
        
        
        q = self.attitude
        self.linear_accel =  quat_rotation_numpy(linear_accel_body, q)
        self.angular_accel = quat_rotation_numpy(angular_accel_body, q)
        #self.linear_accel = linear_accel_body
        #self.angular_accel = angular_accel_body
        self.imu_data = np.concatenate((self.linear_accel, self.angular_accel, self.get_clock().now().nanoseconds), axis=None)
        
        
    

    
    def NED_to_ENU(self, input_array):
        """
        Convert NED frame convention from drone into ENU frame convention used in MPC

        Parameters:
        input_array (np.ndarray): The input to be rotated. Should be a NumPy array of shape (3,) for a 3D vector or (4,) for a quaternion.

        Returns:
        np.ndarray: The rotated 3D vector or quaternion.
        """
        
        if input_array.shape == (3,):
            # Handle as a 3D vector
            # A 180-degree rotation around the x-axis flips the signs of the y and z components
            rot_x = R.from_euler('x', 180, degrees=True)
            rotated_array = rot_x.apply(input_array)
            
            #rotated_array = rot_z.apply(rotated_array)
        elif input_array.shape == (4,):
            
            # Handle as a quaternion
            # A 180-degree rotation around the x-axis flips the signs of the y and z components
            
            input_array = input_array/np.linalg.norm(input_array)
            rotated_array = np.zeros(4)
            
            rotated_array[0] = input_array[0]
            rotated_array[1] = input_array[1]
            rotated_array[2] = -input_array[2]
            rotated_array[3] = -input_array[3]        
            
        else:
            raise ValueError("Input array must be either a 3D vector or a quaternion (shape (3,) or (4,)).")
        
        return rotated_array
    
    
        
    
    def map_thrust(self, input_value):
        """Maps the force requested by MPC to a value between [0,1] for motor command

        Args:
            input_value (float): requested force

        Returns:
            float: mapped motor input
        """
        output = (340.3409*input_value**0.5097284)/1100
        
        if output > 1:
            return 1
        elif output < 0:
            return 0
        else:
            return output
    
    def inverse_map_thrust(self, input_value):
        """Inverse map thrust. Calculate current state of force from motor for MPC. Convert from [%] throttle readout from motors to force.

        Args:
            input_value (int): motor speed. value between [0, 1500] 

        Returns:
            float: current force of motor
        """
        try:
            x = (input_value / 340.3409) ** (1 / 0.5097284)
            
            if 10 < x:
                return 10
            return x
        except ValueError:
            print("Error: The input value of y must be non-negative.")
            return None   
         

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
        
        # correct motor mapping: difference from paper to PX4
        msg.control[0] = control[0]  # Motor 1
        msg.control[1] = control[2]  # Motor 2
        msg.control[2] = control[3]  # Motor 3
        msg.control[3] = control[1]  # Motor 4
        
        self.motor_command_publisher.publish(msg)
        
        
    def publish_motor_command_pseudo(self, control):
        """Publish the motor command setpoint. Pseudo topic, only used for debugging"""
        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        msg.control[0] = control[0]  # Motor 1
        msg.control[1] = control[2]  # Motor 2
        msg.control[2] = control[3]  # Motor 3
        msg.control[3] = control[1]  # Motor 4
        
        self.motor_command_publisher_pseudo.publish(msg)
    
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
        
        
        # wait until enough heartbeat signals have been sent
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
        elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # if in offboard mode: get setpoint from parameters, get optimal U, publish motor command
            
            if self.offboard_setpoint_counter < 100:
                
                # let solver warm up before actually publishing commands
                
                U = self.ocp_solver.solve_for_x0(x0_bar = self.current_state[:-1], fail_on_nonzero_status=False)
                command = np.asarray([self.map_thrust(u) for u in U])
                
                self.publish_motor_command(np.zeros(4))
                self.publish_motor_command_pseudo(command)

            elif self.offboard_setpoint_counter == 100:
                print('starting control')
                self.set_mpc_target_pos()    
            else:      
                
                U = self.ocp_solver.solve_for_x0(x0_bar =  self.current_state[:-1], fail_on_nonzero_status=False)

                command = np.asarray([self.map_thrust(u) for u in U])
                
                # optinally print motor commands
                
                #print('FL: {}, FR: {}'.format(command[3], command[0]))
                #print('BL: {}, BR: {}\n'.format(command[2], command[1]))
                
                
                self.publish_motor_command(command)
                self.publish_motor_command_pseudo(command)
                
                
                
                print("Linear accel: {}".format(self.linear_accel))
                print("Angula accel: {}\n".format(self.angular_accel))
                
                #q = self.attitude
                #lin_accel_rot = quat_rotation_numpy(self.linear_accel, q)
                #ang_accel_rot = quat_rotation_numpy(self.angular_accel, q)
                #
                #print("Lin acc  rot: {}".format(lin_accel_rot))
                #print("Ang acc  rot: {}\n".format(ang_accel_rot))
                
                # optinally print position and attitude
                #print('Position: {}'.format(self.position))
                #print('Velocity: {}'.format(self.velocity))
                #print('Attitude: {}'.format(self.attitude))
                #print('Attitude: {}\n'.format(quaternion_to_euler_numpy(self.attitude)))
            
            
        if self.offboard_setpoint_counter < 200:
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
