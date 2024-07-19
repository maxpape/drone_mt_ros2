#!/usr/bin/env python3

import rclpy
import traceback
from rcl_interfaces.msg import SetParametersResult
import numpy as np
import scipy.linalg
import scipy.interpolate
import time
import collections
import pandas as pd
import GPy
import functions
from casadi import SX, vertcat, Function, sqrt, norm_2, dot, cross, atan2, if_else
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splev, splprep
from scipy.integrate import quad
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleStatus, ActuatorMotors, VehicleOdometry, ActuatorOutputs, SensorCombined, VehicleLocalPosition
from geometry_msgs.msg import Vector3
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange, IntegerRange
from acados_template import AcadosOcp, AcadosOcpSolver
from drone_model import export_drone_ode_model


np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)




def is_setpoint_reached(setpoint, position, attitude, threshold_pos, threshold_att):
    setpoint_position = setpoint[0:3]
    setpoint_attitude = setpoint[3:7]
    
    distance = np.linalg.norm(setpoint_position - position)
    
    attitude_error = functions.quaternion_error_numpy(setpoint_attitude, attitude)
    yaw_error = np.abs(functions.quaternion_to_euler_numpy(attitude_error)[2])
    
    
    
    #if (distance <= threshold_pos) and (yaw_error <= threshold_att):
    if (distance <= threshold_pos):
        return True
    else:
        return False



def generate_trajectory(points, d_points, d_yaw):
    trajectory = []
    for i in range(len(points) - 1):
        vector = points[i + 1][:3] - points[i][:3]
        distance = np.linalg.norm(vector)
        yaw_1 = functions.quaternion_to_euler_numpy(points[i + 1][3:])[2]
        yaw_0 = functions.quaternion_to_euler_numpy(points[i][3:])[2]
        yaw_diff = yaw_1 - yaw_0 
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360
        num_points = max(int(np.ceil(distance / d_points)), int(np.ceil(np.abs(yaw_diff) / d_yaw)))
        for j in range(0, num_points ):
            point = points[i][:3] + j * vector / num_points
            yaw = points[i][3] + j * yaw_diff / num_points
            if yaw > 360:
                yaw -= 360
            elif yaw < -360:
                yaw += 360
            
            trajectory.append(np.concatenate((point, functions.euler_to_quaternion_numpy(np.array([0,0,yaw]))), axis=None))
    trajectory.append(points[-1])
    return np.asarray(trajectory)

def generate_circle_trajectory(start, center, radius, d_points, yaw):
    
    
    n = int(np.ceil(2*np.pi*radius/d_points))+1
    x = center[0]
    y = center[1]
    z = center[2]
    
    #noise = np.random.normal(0, 0.05, (n, 2))
    points = np.array([radius*np.cos(np.linspace(0, 2*np.pi, n))+x,
                       radius*np.sin(np.linspace(0, 2*np.pi, n))+y,
                       np.ones(n)*z]).T
    yaw = np.ones((n-1, 4)) * functions.euler_to_quaternion_numpy(np.array([0,0,yaw]))
    
    return np.hstack((points[:-1], yaw))


def figure_8_arc_length(radius):
    """
    Calculate the arc length of the figure 8 curve.
    
    Parameters:
    radius (float): Radius of one of the loops of the figure 8.
    
    Returns:
    float: The arc length of the figure 8 curve.
    """
    integrand = lambda t: np.sqrt(1 + np.cos(t)**2 + 4 * np.cos(t)**2 * np.sin(t)**2)
    arc_length, _ = quad(integrand, 0, 2 * np.pi)
    return arc_length * radius

def generate_figure_8_trajectory(center, radius, d_points, yaw):
    """
    Generates a 3D figure 8 trajectory with equidistant points.
    
    Parameters:
    radius (float): Radius of one of the loops of the figure 8.
    d (float): Distance between each point on the trajectory.
    
    Returns:
    np.ndarray: Array of shape (N, 3) containing the 3D coordinates of the trajectory.
    """
    x_center = center[0]
    y_center = center[1]
    z_center = center[2]
    
    
    # Calculate the total arc length of the figure 8
    total_length = figure_8_arc_length(radius)
    
    # Determine the number of points needed
    n = int(np.ceil(total_length / d_points))
    
    # Generate points using parametric equations
    t = np.linspace(0, 2 * np.pi, n)
    x = radius * np.cos(t+np.pi/2) 
    y = radius * np.sin(2 * t) / 2 
    #z = np.zeros_like(t)  # Constant height
    z = 0.5 * np.cos(t+np.pi/2)
    
    # Combine into a single array
    points = np.vstack((x+x_center, y+y_center, z+ z_center)).T
    
    
    
    
    yaw = np.ones((n, 4)) * functions.euler_to_quaternion_numpy(np.array([0,0,yaw]))
    return np.hstack((points, yaw))






def find_closest_point(a, b):
    """
    Find the index of the point in array b that is closest to the point in array a.

    Parameters:
    a (numpy.ndarray): A 1D array representing a single point in 3D (shape: (3,))
    b (numpy.ndarray): A 2D array representing multiple points in 3D (shape: (n, 3))

    Returns:
    int: The index of the closest point in array b
    """
    # Calculate the squared differences between the point a and each point in b
    squared_diffs = np.sum((b - a) ** 2, axis=1)
    
    # Find the index of the minimum squared difference
    closest_index = np.argmin(squared_diffs)
    
    return closest_index



    

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
    q_err = functions.quaternion_error_casadi(x[3:7], q_ref)[2]
    
    
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
        
        self.imu_pub_real = self.create_publisher(
            Vector3, '/imu_data_real', qos_profile)
        self.imu_pub_sim = self.create_publisher(
            Vector3, '/imu_data_sim', qos_profile)
        self.imu_pub_gp = self.create_publisher(
            Vector3, '/imu_data_gp', qos_profile)
        self.reference_pub = self.create_publisher(
            Vector3, '/reference_traj', qos_profile)

        # Create subscribers
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_motor_subscriber = self.create_subscription(
            ActuatorOutputs, '/fmu/out/actuator_outputs', self.vehicle_motor_callback, qos_profile)
        

        self.counter = 0
        self.counter_2 = 0
        self.fly_circle = False
        self.init_circle = True
        self.circle_radius = 5.0
        self.circle_center = np.array([0,0,2])
        

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_status = VehicleStatus()
        self.ocp_solver = None
        self.ocp_solver_nominal = None
        
        # variables for ACADOS MPC
        self.N_horizon = 20
        self.Tf = 1
        self.nx = 17
        self.nu = 4
        self.Tmax = 10
        self.Tmin = 1
        self.vmax = 3
        self.angular_vmax = 1.5
        self.max_angle_q = 1
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
        
        # imu data
        history_length = 30
        
        self.linear_accel_real = np.zeros(3)
        self.angular_accel_real = np.zeros(3)
        self.imu_timestamp = 0.0
        self.imu_data = np.concatenate((self.linear_accel_real, np.zeros(3), self.angular_accel_real, np.zeros(3), np.ones(4)*self.hover_thrust, self.get_clock().now().nanoseconds), axis=None)
        self.imu_history = collections.deque(maxlen=history_length)
        
        
        
        self.linear_accel_sim = np.zeros(3)
        self.angular_accel_sim = np.zeros(3)
        self.sim_imu_lin_history = collections.deque(maxlen=history_length)
        self.sim_imu_ang_history = collections.deque(maxlen=history_length)
        
        
        
        
        
        
        #state variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.asarray([np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2])
        self.angular_velocity = np.zeros(3)
        self.thrust = np.ones(4)*self.hover_thrust
        self.state_timestamp = self.get_clock().now().nanoseconds
        self.current_state = np.concatenate((self.position, self.attitude, self.velocity, self.angular_velocity, self.thrust, self.state_timestamp), axis=None)
        self.state_history = collections.deque(maxlen=history_length)
        self.update_current_state()
        
        
        
        
        # state history from MPC sim
        self.state_history_sim = collections.deque(maxlen=history_length)
        
        
        # GP model and parameters
        # gp parameters
        self.lengthscale_lin = np.array([5,5,5])
        self.scale_lin = np.array([0.02, 0.02, 0.02])
        self.noise_variance_lin = 0.25
        
        self.lengthscale_ang = np.array([5,5,5])
        self.scale_ang = np.array([0.02, 0.02, 0.02])
        self.noise_variance_ang = 0.55
        

        
        self.gp_prediction_horizon = 12
        self.gp_multi_step_pred_history = 8
        self.lin_acc_offset = np.zeros((self.gp_prediction_horizon-1,3))
        self.ang_acc_offset = np.zeros((self.gp_prediction_horizon-1,3))
        self.sim_x_last = self.current_state[:-1]
        # prediction history of GP
        self.gp_prediction_history_lin = collections.deque(maxlen=history_length)
        self.gp_prediction_history_ang = collections.deque(maxlen=history_length)
        self.mpc_prediction_history_lin = collections.deque(maxlen=history_length)
        self.mpc_prediction_history_ang = collections.deque(maxlen=history_length)
        self.use_gp = False
        self.offset_applied = False
        self.use_ard = False
        self.online_regression = False
        self.show_lin = True
        self.backsteps_plot = 0
        
        self.length_hypers_lin = [[1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7]]
        self.scale_hypers_lin = [1, 1, 1]
        self.length_hypers_ang = [[1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7]]
        self.scale_hypers_ang = [1, 1, 1]
        
        
        kerns = [GPy.kern.RBF(input_dim=6, variance=self.scale_hypers_lin[0], lengthscale=self.length_hypers_lin[0], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers_lin[0], lengthscale=self.length_hypers_lin[0], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers_lin[0], lengthscale=self.length_hypers_lin[0], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers_lin[0], lengthscale=self.length_hypers_lin[0], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers_lin[0], lengthscale=self.length_hypers_lin[0], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers_lin[0], lengthscale=self.length_hypers_lin[0], active_dims=[0,1,2,3,4,5], ARD=True)]
        lower_lenght = [0.1, 0.1, 4,4,4,4]
        upper_lenght = [10, 10, 20,20,20,20]
        for i in range(6):
            for j in range(6):
                kerns[i].lengthscale[[j]].constrain_bounded(lower_lenght[j], upper_lenght[j])
            #kerns[i].variance.constrain_bounded(1e-3, 5)
            kerns[i].variance.fix()
        
        
        
        self.models = [GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[0]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[1]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[2]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[3]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[4]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[5])]
        
        for model in self.models:
            model.Gaussian_noise.variance = self.noise_variance_lin
            model.Gaussian_noise.variance.fix()
            
        
        
        
        
        # initially fill all ringbuffers
        for x in range(history_length):
            self.imu_history.append(self.imu_data)
            self.sim_imu_lin_history.append(np.array([0,0,0,0,0,0,0,0,0,0,0]))
            self.sim_imu_ang_history.append(np.array([0,0,0,0,0,0,0,0,0,0,0]))
            self.state_history.append(self.current_state)
            self.state_history_sim.append(self.current_state)
            self.gp_prediction_history_lin.append(np.zeros((self.gp_prediction_horizon-1, 3)))
            self.gp_prediction_history_ang.append(np.zeros((self.gp_prediction_horizon-1, 3)))
            self.mpc_prediction_history_lin.append(np.zeros((self.gp_prediction_horizon-1, 6)))
            self.mpc_prediction_history_ang.append(np.zeros((self.gp_prediction_horizon-1, 6)))
        
        #setpoint variables
        self.position_setpoint = np.array([0,0,2])
        self.velocity_setpoint = np.zeros(3)
        self.attitude_setpoint = np.asarray([np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2])
        self.roll_setpoint = 0
        self.pitch_setpoint = 0
        self.yaw_setpoint = 0
        self.angular_velocity_setpoint = np.zeros(3)
        self.setpoint = np.concatenate((self.position_setpoint, self.attitude_setpoint), axis=None)
        self.setpoints = []
        self.trajectory = []
        self.update_setpoint()
        

        


        


        self.parameters = np.concatenate((self.params, self.setpoint, np.zeros(6)), axis=None)

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
        
        
        # Declare a parameter with a descriptor for dynamic reconfiguration
        gp_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,  # Specify the type as double
            description='Desired parameter',     # Description of the parameter
            additional_constraints='Range: 0.001 to 30',
            floating_point_range=[FloatingPointRange(
                from_value=0.001,  # Minimum value
                to_value=30.0,     # Maximum value
                step=0.001         # Step size (optional)
            )]# Constraints (optional)
        )
        
        self.declare_parameters(
        namespace='',
        parameters=[
            ('lengthscale_lin', 5.0, gp_descriptor),
            ('scale_lin', 0.2, gp_descriptor),
            ('noise_variance_lin', 0.01, gp_descriptor),
            ('lengthscale_ang', 5.0, gp_descriptor),
            ('scale_ang', 0.2, gp_descriptor),
            ('noise_variance_ang', 0.01, gp_descriptor),
        ]
        )
        
        
        int_range = IntegerRange(from_value=0, to_value=self.gp_prediction_horizon-2, step=1)
        
        # Create a parameter descriptor with the integer range
        int_range_descriptor = ParameterDescriptor(integer_range=[int_range])
 
        
        self.declare_parameters(
        namespace='',
        parameters=[
            ('use_gp', False),
            ('use_ard', False),
            ('online_regression', False),
            ('backsteps_plot', 0, int_range_descriptor),
            ('show_lin', True),
            ('fly_circle', False) 
        ]
        )
        
        
        
        self.add_on_set_parameters_callback(self.parameter_callback)
        
    def update_current_state(self):
        """aggregates individual states to combined state of system
        """
        
        self.current_state = np.concatenate((self.position, self.attitude, self.velocity, self.angular_velocity, self.thrust, self.state_timestamp), axis=None)

        #self.imu_data = np.concatenate((self.linear_accel_real, self.angular_accel_real, self.imu_timestamp), axis=None)
        
        self.state_history.append(self.current_state)
        #self.imu_history.append(self.imu_data) 
        
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
            self.attitude_setpoint = functions.euler_to_quaternion_numpy(np.array([self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint]))
        if "pitch" in fields:
            self.pitch_setpoint = pitch
            self.attitude_setpoint = functions.euler_to_quaternion_numpy(np.array([self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint]))
        if "yaw" in fields:
            self.yaw_setpoint = yaw
            self.attitude_setpoint = functions.euler_to_quaternion_numpy(np.array([self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint]))
        
        self.setpoint = np.concatenate((self.position_setpoint, self.attitude_setpoint), axis=None)
        self.setpoints.append(self.setpoint)
        
    
    def set_mpc_target_pos(self):
        
        
        dist_points = self.vmax/self.N_horizon*self.Tf
        
        if (len(self.setpoints) > 1):
            
            for i in range(len(self.setpoints)):
                
                reached = is_setpoint_reached(self.setpoints[i], self.current_state[0:3], self.attitude, dist_points*2, 11)
                if reached:
                    self.setpoints = self.setpoints[i+1:]
                    
                    break
        
        
        
        
        if not self.fly_circle:
            start = np.concatenate((self.position, self.attitude), axis=None)
            self.trajectory = generate_trajectory(np.vstack((start, self.setpoints)), dist_points*0.65, 10)
            
            reference = Vector3()
              
            reference.x = self.setpoints[0][0]
            reference.y = self.setpoints[0][1]
            reference.z = self.setpoints[0][2]
            self.reference_pub.publish(reference) 
            
        else:
            if self.init_circle:
                start = self.current_state[0:3]
                self.circle_center = start + np.array([-self.circle_radius, 0, 0])
                yaw = functions.quaternion_to_euler_numpy(self.attitude)[2]
                
                #self.trajectory = generate_circle_trajectory(start, self.circle_center, self.circle_radius, dist_points*0.65, yaw)
                self.trajectory = generate_figure_8_trajectory(start, self.circle_radius, dist_points*0.65, yaw)
                self.init_circle = False
                
            #if self.circle_traj_index >= len(self.trajectory):
            #    self.circle_traj_index = 0
            
            #reached = is_setpoint_reached(self.trajectory[0], self.current_state[0:3], self.attitude, dist_points*0.70, 11)
            
                #self.circle_traj_index += i
            reference = Vector3()
              
            reference.x = self.trajectory[0][0]
            reference.y = self.trajectory[0][1]
            reference.z = self.trajectory[0][2]
            self.reference_pub.publish(reference) 
            
            a, b = self.trajectory[1:], self.trajectory[0]
            self.trajectory = np.vstack((a, b))
                
                
            
        
            
            
                
        
        
        if len(self.trajectory) <= self.N_horizon:
            
            last_elem = self.trajectory[-1]
            dim = self.N_horizon - len(self.trajectory)
            
            filler = np.full((dim+2,7), last_elem)
            
            trajectory = np.vstack((self.trajectory , filler))
        
        if len(self.trajectory) > self.N_horizon:
            trajectory = self.trajectory[0:self.N_horizon+2]
            
        
                
        
        parameters = np.concatenate((self.params, trajectory[0], np.zeros(6)), axis=None)
        self.ocp_solver.set(0, "p", parameters) 
        self.ocp_solver_nominal.set(0, "p", parameters)  
        
        
        
        
        
        
        for j in range(1, self.gp_prediction_horizon-1):
            if self.use_gp:
                
                parameters = np.concatenate((self.params, trajectory[j], self.lin_acc_offset[j], self.ang_acc_offset[j]), axis=None)
                parameters_nominal = np.concatenate((self.params, trajectory[j],  np.zeros(6)), axis=None)
                self.offset_applied = True
                
                
                    
            else:
                parameters = np.concatenate((self.params, trajectory[j],  np.zeros(6)), axis=None)
                parameters_nominal = np.concatenate((self.params, trajectory[j],  np.zeros(6)), axis=None)
                self.offset_applied = False
                 
            self.ocp_solver_nominal.set(j, "p", parameters_nominal) 
            self.ocp_solver.set(j, "p", parameters)
            
        for j in range(self.gp_prediction_horizon-1, self.N_horizon):
            parameters = np.concatenate((self.params, trajectory[j], np.zeros(6)), axis=None)
            
            self.ocp_solver.set(j, "p", parameters)
            self.ocp_solver_nominal.set(j, "p", parameters)
        
        parameters = np.concatenate((self.params, trajectory[self.N_horizon], np.zeros(6)), axis=None)
        self.ocp_solver.set(self.N_horizon, "p", parameters)
        self.ocp_solver_nominal.set(self.N_horizon, "p", parameters)  
    
    
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
            elif param.name == "lengthscale_lin":
                self.lengthscale_lin[:] = param.value
            elif param.name == "scale_lin":
                self.scale_lin[:] = param.value
            elif param.name == "noise_variance_lin":
                self.noise_variance_lin = param.value
            elif param.name == "lengthscale_ang":
                self.lengthscale_ang[:] = param.value
            elif param.name == "scale_ang":
                self.scale_ang[:] = param.value
            elif param.name == "noise_variance_ang":
                self.noise_variance_ang = param.value
            elif param.name == "use_gp":
                self.use_gp = param.value
            elif param.name == "use_ard":
                self.use_ard = param.value
            elif param.name == "online_regression":
                self.online_regression = param.value
                #self.lengthscale = np.array([2, 2, 2, 0.01, 0.01, 0.01])
                #self.scale = np.array([1, 1, 1, 0.8, 0.8, 0.8])
            elif param.name == "backsteps_plot":
                self.backsteps_plot = param.value
            elif param.name == "show_lin":
                self.show_lin = param.value
            elif param.name == "fly_circle":
                if param.value:
                    self.fly_circle = True
                    self.init_circle = True
                else:
                    self.fly_circle = False
                    self.init_circle = False
                    
         
               
        self.attitude_setpoint = functions.euler_to_quaternion_numpy(np.array([self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint]))        
        
        self.update_setpoint()
        print('New target setpoint: {}'.format(self.setpoint))
        
        self.set_mpc_target_pos()
        
        return SetParametersResult(successful=True)

    
    def predict_accel_lin(self, x, y, new_x, axis, dim=6):
        
        
            
        self.models[axis].set_XY(x, y)
        
        for i in range(dim):
            self.models[axis].rbf.lengthscale[i] = self.length_hypers_lin[axis][i] 
        self.models[axis].rbf.variance[0] = self.scale_hypers_lin[axis] 
        self.models[axis].Gaussian_noise.variance = self.noise_variance_lin
        
        
        if self.online_regression :
            self.models[axis].optimize(max_iters=1)
            
            
            
        for i in range(dim):
            self.length_hypers_lin[axis][i] = self.models[axis].rbf.lengthscale[i]
        
        self.scale_hypers_lin[axis] = self.models[axis].rbf.variance[0]
        
        if axis == 0 and self.show_lin:
            print(self.length_hypers_lin[axis])
            print(self.scale_hypers_lin[axis])
            print('-------------')
        mean, var = self.models[axis].predict(new_x)
        
        
        
            
        return mean, var
    
    
    def predict_accel_ang(self, x, y, new_x, axis, dim=6):
        
        
        self.models[axis+3].set_XY(x, y)
        
        for i in range(dim):
            self.models[axis+3].rbf.lengthscale[i] = self.length_hypers_ang[axis][i] 
        self.models[axis+3].rbf.variance[0] = self.scale_hypers_lin[axis] 
        self.models[axis+3].Gaussian_noise.variance = self.noise_variance_ang
        self.models[axis+3].Gaussian_noise.variance.fix()
        
        
        if self.online_regression :
            self.models[axis+3].optimize(max_iters=1)
         
            
            
            
            
        for i in range(dim):
            self.length_hypers_ang[axis][i] = self.models[axis+3].rbf.lengthscale[i]
        
        self.scale_hypers_ang[axis] = self.models[axis].rbf.variance[0]
        
        if axis == 0 and (not self.show_lin):
            print(self.length_hypers_ang[axis])
            print(self.scale_hypers_ang[axis])
            print('-------------')
        mean, var = self.models[axis+3].predict(new_x)
            
            
          
            
        
        return mean, var


    
    def predict_next_y(self, x, y, new_x, type):
        """
        Predict the next y for a given x using GPy.

        Parameters:
        x (numpy array): Input data from the last 20 timesteps.
        y (numpy array): Output data from the last 20 timesteps.
        new_x (numpy array): New input for which to predict the output.

        Returns:
        numpy array: Predicted output for the new input.
        """
        
        
        if type == 0:
            kern = GPy.kern.RBF(input_dim=2, variance=self.scale_lin[0], lengthscale=self.lengthscale_lin[0], active_dims=[0,1], ARD=self.use_ard)
            #kern.variance.constrain_bounded(0.1, 1, warning=False)  # Set variance bounds
            #kern.lengthscale.constrain_bounded(1, 7, warning=False)  # Set lengthscale bounds
            gpmodel = GPy.models.GPRegression(x, y, kern)
            gpmodel.Gaussian_noise.variance = self.noise_variance_lin
            gpmodel.Gaussian_noise.variance.fix()
            
        elif type == 1:
            kern = GPy.kern.RBF(input_dim=2, variance=self.scale_lin[1], lengthscale=self.lengthscale_lin[1], active_dims=[0,1], ARD=self.use_ard)
            #kern.variance.constrain_bounded(0.1, 1, warning=False)  # Set variance bounds
            #kern.lengthscale.constrain_bounded(1, 7, warning=False)  # Set lengthscale bounds
            gpmodel = GPy.models.GPRegression(x, y, kern)
            gpmodel.Gaussian_noise.variance = self.noise_variance_lin
            gpmodel.Gaussian_noise.variance.fix()
        
        elif type == 3:
            kern = GPy.kern.RBF(input_dim=2, variance=self.scale_ang[0], lengthscale=self.lengthscale_ang[0], active_dims=[0,1], ARD=self.use_ard)
            #kern.variance.constrain_bounded(0.1, 1, warning=False)  # Set variance bounds
            #kern.lengthscale.constrain_bounded(4, 7, warning=False)  # Set lengthscale bounds
            gpmodel = GPy.models.GPRegression(x, y, kern)
            gpmodel.Gaussian_noise.variance = self.noise_variance_ang
            gpmodel.Gaussian_noise.variance.fix() 
            
        elif type == 4:
            kern = GPy.kern.RBF(input_dim=2, variance=self.scale_ang[1], lengthscale=self.lengthscale_ang[1], active_dims=[0,1], ARD=self.use_ard)
            #kern.variance.constrain_bounded(0.1, 1, warning=False)  # Set variance bounds
            #kern.lengthscale.constrain_bounded(4, 7, warning=False)  # Set lengthscale bounds
            gpmodel = GPy.models.GPRegression(x, y, kern)
            gpmodel.Gaussian_noise.variance = self.noise_variance_ang
            gpmodel.Gaussian_noise.variance.fix() 
            
        elif type == 5:
            kern = GPy.kern.RBF(input_dim=2, variance=self.scale_ang[2], lengthscale=self.lengthscale_ang[2], active_dims=[0,1], ARD=self.use_ard)
            #kern.variance.constrain_bounded(0.1, 1, warning=False)  # Set variance bounds
            #kern.lengthscale.constrain_bounded(4, 7, warning=False)  # Set lengthscale bounds
            gpmodel = GPy.models.GPRegression(x, y, kern)
            gpmodel.Gaussian_noise.variance = self.noise_variance_ang
            gpmodel.Gaussian_noise.variance.fix()   
        
        #rbf_kern.variance.constrain_bounded(0.05, 0.3, warning=False)  # Set variance bounds
        #rbf_kern.lengthscale.constrain_bounded(0.04, 0.07, warning=False)  # Set lengthscale bounds
        
        
        # Define a non-zero mean function
        #x_mean = np.mean(x, axis=0)[0]
        #y_mean = np.mean(y)
        ######
        #data_mean = np.array([0,0])
        ##if print_mean:
        ##    print(data_mean)
        #mean_function = GPy.mappings.Constant(input_dim=2, output_dim=1, value=data_mean)

        
        # Optimize the model parameters
        
            
            
            
            
            #if type == 'ang' and axis == 0:
            #    print(gpmodel.rbf.lengthscale)
                #print(int(axis+type_dict[type]))
                #print('Lengthscale: {}'.format(self.lengthscale[int(axis+type_dict[type])]))
                #print('Variance: {}'.format(self.scale[int(axis+type_dict[type])]))
            
                
         
        #if axis == 2:   
        #    print('variance: {}'.format(self.scale[2]))
        #    print('lengthscale: {}'.format(self.lengthscale[2]))
        #    print('\n')   
        
        
        #print('bias variance: {}'.format(self.gpmodel.sum.bias.variance[0]))
        #if self.counter == 30:
        #    self.counter = 0
        #print('lengthscale: {}'.format(gpmodel.sum.rbf.lengthscale[0]))
        #print('variance: {}'.format(gpmodel.sum.rbf.variance[0]))
        #print('variance after optimization: {}'.format(model.linear.variances[0]))


        #self.scale = self.gpmodel.rbf.variance[0]
        #self.lengthscale = self.gpmodel.rbf.lengthscale[0]
        

        # Predict the mean and variance of the output for the new input
        mean, var = gpmodel.predict(new_x)
        
        # Return the predicted mean
        return mean, var
    
    
    
    
       
    def setup_mpc(self):
        
        ocp = AcadosOcp()
        
        # set model
        model = export_drone_ode_model()
        ocp.model = model
        

        ocp.dims.N = self.N_horizon
        ocp.parameter_values = self.parameters
        
        
        # define weighing matrices
        Q_p= np.diag([40,40,200])*5
        Q_q= np.eye(1)*100
        Q_mat = scipy.linalg.block_diag(Q_p, Q_q)
    
        R_U = np.eye(4)
        
        Q_p_final = np.diag([28,28,200])*30
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
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_OSQP'
        
        
        # create ACADOS solver
        solver_json = 'acados_ocp_' + model.name + '.json'
        self.ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)
        self.ocp_solver_nominal = AcadosOcpSolver(ocp, json_file = solver_json)
        
        
        
        
        



    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
        
    def vehicle_odometry_callback(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber.
        Updates position, velocity, attitude and angular velocity

        Args:
            vehicle_odometry (px4_msgs.msg.VehicleOdometry): odometry message from drone
        """
        
        self.state_timestamp = vehicle_odometry.timestamp
        self.position = functions.NED_to_ENU(vehicle_odometry.position)
        self.velocity = functions.NED_to_ENU(vehicle_odometry.velocity)
        self.attitude = functions.NED_to_ENU(vehicle_odometry.q)
        self.angular_velocity = functions.NED_to_ENU(vehicle_odometry.angular_velocity)
        
        
        
              
    
    
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
        
  
    
    def map_thrust(self, input_value):
        """Maps the force requested by MPC to a value between [0,1] for motor command

        Args:
            input_value (float): requested force

        Returns:
            float: mapped motor input
        """
        output = (341.2561*input_value**0.5004472)/1100
        
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
            x = (input_value / 341.2561) ** (1 / 0.5004472)
            
            if 10 < x:
                return 10
            return x
        except ValueError:
            print("Error: The input value of y must be non-negative.")
            return None   
    
    
    
    def calculate_sim_acc_from_vel(self, simx, lin_or_ang='lin'):
        
        vel = {"lin": simx[:,7:10],
                    "ang": simx[:,10:13],
                    }
        
        offset = {"lin": self.lin_acc_offset,
                    "ang": self.ang_acc_offset,
                    }
        history = {"lin": self.mpc_prediction_history_lin,
                    "ang": self.mpc_prediction_history_ang,
                    }
        
        
        velocity = vel[lin_or_ang]
        
        # if GP is used, substract acceleration offset added in model from predicted velocity
        
        
        
        
        
        v0 = velocity[:-1]
        v1 = velocity[1:]
        
       
        accel = (v1 - v0) / (self.Tf/self.N_horizon)
        
        #if self.use_gp and self.offset_applied:
        #    pad = np.zeros(3)
        #    correction = np.vstack((pad, offset[lin_or_ang][1:])) 
        #    accel = accel  - correction / 20
        
            
            
        accel_vel = np.hstack((accel, v1, simx[1:,13:17]))
        
        history[lin_or_ang].append(accel_vel)
            
        return accel_vel
        
    def publish_for_plotjuggler(self, real_hist, is_linear=True):
        
        mpc_pred = [self.mpc_prediction_history_lin, self.mpc_prediction_history_ang]
        gp_pred = [self.gp_prediction_history_lin, self.gp_prediction_history_ang]
        
        # Determine index based on the boolean flag
        index = 0 if is_linear else 1
        
        imu_real = Vector3()
        imu_sim = Vector3()
        imu_gp = Vector3()
        backsteps = self.backsteps_plot

        # Assign real history values
        imu_real.x = real_hist[-1-backsteps][0]
        imu_real.y = real_hist[-1-backsteps][1]
        imu_real.z = real_hist[-1-backsteps][2]

        ## Calculate errors
        #x_error = gp_pred[index][-2-backsteps][backsteps, 0] - mpc_pred[index][-2-backsteps][backsteps, 0]
        #y_error = gp_pred[index][-2-backsteps][backsteps, 1] - mpc_pred[index][-2-backsteps][backsteps, 1]
        #z_error = gp_pred[index][-2-backsteps][backsteps, 2] - mpc_pred[index][-2-backsteps][backsteps, 2]
        
        
        # Calculate errors
        x_error = gp_pred[index][-2-backsteps][backsteps, 0] 
        y_error = gp_pred[index][-2-backsteps][backsteps, 1] 
        z_error = gp_pred[index][-2-backsteps][backsteps, 2] 
        

        # Assign simulated values
        imu_sim.x = float(mpc_pred[index][-2-backsteps][backsteps, 0])
        imu_sim.y = float(mpc_pred[index][-2-backsteps][backsteps, 1])
        imu_sim.z = float(mpc_pred[index][-2-backsteps][backsteps, 2])

        # Assign GP values
        imu_gp.x = float(mpc_pred[index][-2-backsteps][backsteps, 0] + x_error)
        imu_gp.y = float(mpc_pred[index][-2-backsteps][backsteps, 1] + y_error)
        imu_gp.z = float(mpc_pred[index][-2-backsteps][backsteps, 2] + z_error)

        # Publish the values
        self.imu_pub_gp.publish(imu_gp)
        self.imu_pub_real.publish(imu_real)
        self.imu_pub_sim.publish(imu_sim)

    
    

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
        msg.control[1] = control[1]  # Motor 2
        msg.control[2] = control[2]  # Motor 3
        msg.control[3] = control[3]  # Motor 4
        
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
        
        start = time.time()
        self.publish_offboard_control_heartbeat_signal()
        
        self.update_current_state()
         
        # wait until enough heartbeat signals have been sent
        
        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # if in offboard mode: get setpoint from parameters, get optimal U, publish motor command
            
            if self.offboard_setpoint_counter < 100:
                
                # let solver warm up before actually publishing commands
                
                U = self.ocp_solver.solve_for_x0(x0_bar = self.current_state[:-1], fail_on_nonzero_status=False)
                command = np.asarray([self.map_thrust(u) for u in U])
                
                self.publish_motor_command(np.zeros(4))
               
                #U = self.ocp_solver_nominal.solve_for_x0(x0_bar = self.current_state[:-1], fail_on_nonzero_status=False)
                # get simulated response for next #prediction_horizon time steps
                simx = []
                for i in range(self.gp_prediction_horizon):
                    simx.append(self.ocp_solver.get(i, 'x'))
                simx = np.asarray(simx)
                
                
                # calculate linear acceleration for next time steps; used to predict error
                sim_accel_pred_lin = self.calculate_sim_acc_from_vel(simx, 'lin')
                
                
                # calculate angular acceleration for next time steps; used to predict error
                sim_accel_pred_ang = self.calculate_sim_acc_from_vel(simx, 'ang')

                
                
                
                # append calculated 1-step-prediction for acceleration to history ringbuffer
                sim_accel_lin = sim_accel_pred_lin[0]
                sim_accel_ang = sim_accel_pred_ang[0]
                t = self.get_clock().now().nanoseconds
                self.sim_imu_lin_history.append(np.concatenate((sim_accel_lin, t), axis=None))
                self.sim_imu_ang_history.append(np.concatenate((sim_accel_ang, t), axis=None))
                
                
                
                # now for the real acceleration
                # extract linear and angular velocity
                realx_0_v = self.state_history[-2][7:10]
                realx_1_v = self.state_history[-1][7:10]
                realx_0_w = self.state_history[-2][10:13]
                realx_1_w = self.state_history[-1][10:13]
                
                
                
                # get timestamps from states
                t0 = self.state_history[-2][-1] * 1e-6
                t1 = self.state_history[-1][-1] * 1e-6
                dt = (t1-t0)
                if dt == 0:
                    dt = self.Tf/self.N_horizon
                
                # calculate acceleration from velocity
                real_accel_lin = (realx_1_v - realx_0_v) / (dt)
                real_accel_ang = (realx_1_w - realx_0_w) / (dt)
                
                
                # append calculated acceleration to history ringbuffer
                
                
                self.linear_accel_real = real_accel_lin
                self.angular_accel_real = real_accel_ang
                self.imu_data = np.concatenate((self.linear_accel_real, realx_1_v, self.angular_accel_real, realx_1_w, self.state_history[-1][13:17],  self.current_state[-1]), axis=None)
                
                self.imu_history.append(self.imu_data)
                
         
                

                  
            else:
                # set parameters and trajectory     
                self.set_mpc_target_pos()
                # solve OCP and publish motor command
                U = self.ocp_solver.solve_for_x0(x0_bar =  self.current_state[:-1], fail_on_nonzero_status=False)
                command = np.asarray([self.map_thrust(u) for u in U])
                self.publish_motor_command(command)
                
                U = self.ocp_solver_nominal.solve_for_x0(x0_bar = self.current_state[:-1], fail_on_nonzero_status=False)
                # get simulated response for next #prediction_horizon time steps
                simx = []
                for i in range(self.gp_prediction_horizon):
                    x = self.ocp_solver_nominal.get(i, 'x')
                    
                    #simx.append(np.hstack((x,u)))
                    simx.append(x)
                simx = np.asarray(simx)
                
                
                # calculate linear acceleration for next time steps; used to predict error
                sim_accel_pred_lin = self.calculate_sim_acc_from_vel(simx, 'lin')
                
                
                # calculate angular acceleration for next time steps; used to predict error
                sim_accel_pred_ang = self.calculate_sim_acc_from_vel(simx, 'ang')

                
                
                
                # append calculated 1-step-prediction for acceleration to history ringbuffer
                sim_accel_lin = sim_accel_pred_lin[0]
                sim_accel_ang = sim_accel_pred_ang[0]
                t = self.get_clock().now().nanoseconds
                self.sim_imu_lin_history.append(np.concatenate((sim_accel_lin, t), axis=None))
                self.sim_imu_ang_history.append(np.concatenate((sim_accel_ang, t), axis=None))
                
                # now for the real acceleration
                # extract linear and angular velocity
                realx_0_v = self.state_history[-2][7:10]
                realx_1_v = self.state_history[-1][7:10]
                realx_0_w = self.state_history[-2][10:13]
                realx_1_w = self.state_history[-1][10:13]
              
                # get timestamps from states
                t0 = self.state_history[-2][-1] * 1e-6
                t1 = self.state_history[-1][-1] * 1e-6
                
                # calculate acceleration from velocity
                dt = (t1-t0)
                if dt == 0:
                    dt = self.Tf/self.N_horizon
                real_accel_lin = (realx_1_v - realx_0_v) / (dt)
                real_accel_ang = (realx_1_w - realx_0_w) / (dt)
                
                # append calculated acceleration to history ringbuffer
                self.linear_accel_real = real_accel_lin
                self.angular_accel_real = real_accel_ang
                self.imu_data = np.concatenate((self.linear_accel_real, realx_1_v, self.angular_accel_real, realx_1_w, self.state_history[-1][13:17], self.current_state[-1]), axis=None)
                self.imu_history.append(self.imu_data)

                
                
                
                
             
                ## prediction of linear acceleration error
                real_hist_lin = np.nan_to_num(np.asarray(list(self.imu_history)[1:]), nan=0)[:,(0,1,2,3,4,5,6,13,14,15,16)]             # history of measured acceleration and velocity
                #sim_hist_lin = np.nan_to_num(np.asarray(list(self.sim_imu_lin_history)[:-1]), nan=0)     # history of acceleration and velocity from 1-step prediction of mpc
                sim_accel_pred_lin_ext = np.vstack((sim_accel_pred_lin, self.sim_imu_lin_history[-2][:-1]))     # multi-step prediction of acceleration and velocity from mpc
                #
                #error = real_hist_lin - sim_hist_lin
                
                
                
                
                # prepare linear prediction
                hist_sim_lin_x = self.mpc_prediction_history_lin[-(self.gp_multi_step_pred_history)][:self.gp_multi_step_pred_history-1,(0,3,6,7,8,9)] 
                hist_real_lin_x = np.nan_to_num(np.asarray(list(self.imu_history)[-(self.gp_multi_step_pred_history-1):])[:,(0,3,12,13,14,15)], nan=0) 
                error_lin_x = hist_real_lin_x - hist_sim_lin_x
                
                hist_sim_lin_y = self.mpc_prediction_history_lin[-(self.gp_multi_step_pred_history)][:self.gp_multi_step_pred_history-1,(1,4,6,7,8,9)]
                hist_real_lin_y = np.nan_to_num(np.asarray(list(self.imu_history)[-(self.gp_multi_step_pred_history-1):])[:,(1,4,12,13,14,15)], nan=0)
                error_lin_y = hist_real_lin_y - hist_sim_lin_y
                
                hist_sim_lin_z = self.mpc_prediction_history_lin[-(self.gp_multi_step_pred_history)][:self.gp_multi_step_pred_history-1,(2,5,6,7,8,9)]
                hist_real_lin_z = np.nan_to_num(np.asarray(list(self.imu_history)[-(self.gp_multi_step_pred_history-1):])[:,(2,5,6,7,8,9)], nan=0)
                error_lin_z = hist_real_lin_z - hist_sim_lin_z

                
                
                #certainty_x = 1 - (gp_prediction_lin_x_var / np.max(gp_prediction_lin_x_var))
                #gp_prediction_lin_x = np.multiply(gp_prediction_lin_x, certainty_x)
          
                
                
                
                
                

              

                
                # calculate offset from prediction of GP and MPC
                # lin_acc_offset: estimated real acceleration from GP
                
                
               
                
                
                
                
                # prediction of angular acceleration error
                # prepare input data for GP
                real_hist_ang = np.nan_to_num(np.asarray(list(self.imu_history)[1:])[:, 6:12], nan=0)
                #sim_hist_ang = np.nan_to_num(np.asarray(list(self.sim_imu_ang_history)[:-1])[:, 0:6], nan=0)
                sim_accel_pred_ang_ext = np.vstack((sim_accel_pred_ang, self.sim_imu_ang_history[-2][:-1]))
                

                # prepare angular prediction
                hist_sim_ang_x = self.mpc_prediction_history_ang[-(self.gp_multi_step_pred_history)][:self.gp_multi_step_pred_history-1,(0,3,6,7,8,9)]
                hist_real_ang_x = np.nan_to_num(np.asarray(list(self.imu_history)[-(self.gp_multi_step_pred_history-1):])[:,(6,9,12,13,14,15)], nan=0)
                error_ang_x = hist_real_ang_x - hist_sim_ang_x

                hist_sim_ang_y = self.mpc_prediction_history_ang[-(self.gp_multi_step_pred_history)][:self.gp_multi_step_pred_history-1,(1,4,6,7,8,9)]
                hist_real_ang_y = np.nan_to_num(np.asarray(list(self.imu_history)[-(self.gp_multi_step_pred_history-1):])[:,(7,10,12,13,14,15)], nan=0)
                error_ang_y = hist_real_ang_y - hist_sim_ang_y
                
                hist_sim_ang_z = self.mpc_prediction_history_ang[-(self.gp_multi_step_pred_history)][:self.gp_multi_step_pred_history-1,(2,5,6,7,8,9)]
                hist_real_ang_z = np.nan_to_num(np.asarray(list(self.imu_history)[-(self.gp_multi_step_pred_history-1):])[:,(8,11,12,13,14,15)], nan=0)
                error_ang_z = hist_real_ang_z - hist_sim_ang_z
               
                
                start = time.time()
                # prepare prediction args for multiprocessing
                
                
                #prediction_args0 = [(self, np.asarray(hist_sim_lin_x), np.asarray(error_lin_x[:,0].reshape(-1,1)), np.asarray(sim_accel_pred_lin_ext[:,(0,3,6,7,8,9)]), 0),
                #                    (self, np.asarray(hist_sim_lin_y), np.asarray(error_lin_y[:,0].reshape(-1,1)), np.asarray(sim_accel_pred_lin_ext[:,(1,4,6,7,8,9)]), 1),
                #                    (self, np.asarray(hist_sim_lin_z), np.asarray(error_lin_z[:,0].reshape(-1,1)), np.asarray(sim_accel_pred_lin_ext[:,(2,5,6,7,8,9)]), 2)]
                #                   
                #                   
                #prediction_args1 = [(self, np.asarray(hist_sim_lin_x), np.asarray(error_ang_x[:,0].reshape(-1,1)), np.asarray(sim_accel_pred_ang_ext[:,(0,3,6,7,8,9)]), 0),
                #                    (self, np.asarray(hist_sim_lin_y), np.asarray(error_ang_y[:,0].reshape(-1,1)), np.asarray(sim_accel_pred_ang_ext[:,(1,4,6,7,8,9)]), 1),
                #                    (self, np.asarray(hist_sim_lin_z), np.asarray(error_ang_z[:,0].reshape(-1,1)), np.asarray(sim_accel_pred_ang_ext[:,(2,5,6,7,8,9)]), 2)]
                
                
                #prediction_args0 = [(np.ones((6,12)), np.ones((6,1)), np.ones((6,12)), 0),
                #                    (np.ones((6,12)), np.ones((6,1)), np.ones((6,12)), 1),
                #                    (np.ones((6,12)), np.ones((6,1)), np.ones((6,12)), 2)]
                #                   
                #                   
                #prediction_args1 = [(np.ones((6,12)), np.ones((6,1)), np.ones((6,12)), 0),
                #                    (np.ones((6,12)), np.ones((6,1)), np.ones((6,12)), 1),
                #                    (np.ones((6,12)), np.ones((6,1)), np.ones((6,12)), 2)]
                
                
               
        
               
                #with Pool(processes=3) as pool:
                #    results0 = pool.map(predict_accel_lin_wrapper, prediction_args0)
                #with Pool(processes=3) as pool:
                #    results1 = pool.map(predict_accel_ang_wrapper, prediction_args1)
                
                
                # predict all errors
                gp_prediction_lin_x, gp_prediction_lin_x_var = self.predict_accel_lin(hist_sim_lin_x, error_lin_x[:,0].reshape(-1,1), sim_accel_pred_lin_ext[:,(0,3,6,7,8,9)], axis=0)
                gp_prediction_lin_y, gp_prediction_lin_y_var = self.predict_accel_lin(hist_sim_lin_y, error_lin_y[:,0].reshape(-1,1), sim_accel_pred_lin_ext[:,(1,4,6,7,8,9)], axis=1)
                gp_prediction_lin_z, gp_prediction_lin_z_var = self.predict_accel_lin(hist_sim_lin_z, error_lin_z[:,0].reshape(-1,1), sim_accel_pred_lin_ext[:,(2,5,6,7,8,9)], axis=2) 
                gp_prediction_ang_x, gp_prediction_ang_x_var = self.predict_accel_ang(hist_sim_ang_x, error_ang_x[:,0].reshape(-1,1), sim_accel_pred_ang_ext[:,(0,3,6,7,8,9)], axis=0)
                gp_prediction_ang_y, gp_prediction_ang_y_var = self.predict_accel_ang(hist_sim_ang_y, error_ang_y[:,0].reshape(-1,1), sim_accel_pred_ang_ext[:,(1,4,6,7,8,9)], axis=1)
                gp_prediction_ang_z, gp_prediction_ang_z_var = self.predict_accel_ang(hist_sim_ang_z, error_ang_z[:,0].reshape(-1,1), sim_accel_pred_ang_ext[:,(2,5,6,7,8,9)], axis=2)

                
                
                
                
                
                #gp_prediction_lin_x = np.zeros((self.gp_prediction_horizon, 1))
                #gp_prediction_lin_y = np.zeros((self.gp_prediction_horizon, 1))
                #gp_prediction_lin_z = np.zeros((self.gp_prediction_horizon, 1))
                #gp_prediction_ang_x = np.zeros((self.gp_prediction_horizon, 1))
                #gp_prediction_ang_y = np.zeros((self.gp_prediction_horizon, 1))
                #gp_prediction_ang_z = np.zeros((self.gp_prediction_horizon, 1))
                
                
                # sace prediction results
                lin_acc_offset = np.hstack((gp_prediction_lin_x[:-1,0].reshape(-1,1), gp_prediction_lin_y[:-1,0].reshape(-1,1), gp_prediction_lin_z[:-1,0].reshape(-1,1)))
                self.lin_acc_offset = lin_acc_offset.clip(min=-10, max=10)
                self.gp_prediction_history_lin.append(lin_acc_offset)
                
                ang_acc_offset = np.hstack((gp_prediction_ang_x[:-1,0].reshape(-1,1), gp_prediction_ang_y[:-1,0].reshape(-1,1), gp_prediction_ang_z[:-1,0].reshape(-1,1)))
                self.ang_acc_offset = ang_acc_offset.clip(min=-10, max=10)
                self.gp_prediction_history_ang.append(ang_acc_offset)
                
                
                
                
                
                
                
                
                
                self.counter += 1
                if self.counter == 6:
                    self.counter = 0
                #
                
                
                
                
                # calculate offset from prediction of GP and MPC
                
                
                
                
                
                #postition_hist = np.asarray(list(self.state_history))[:,0:3]    
                #postition_average = np.mean(postition_hist, axis=0)
                #
                #position_error = self.setpoint[0:3] - postition_average
                
                #print('Average deviation: {}'.format(position_error))
                
                
                
                
                # publish data to compare: real acceleration vs. simulated acceleration from mpc and predicted acceleration form GP
                # publish data for plotjuggler
                
                if self.show_lin:
                    self.publish_for_plotjuggler(real_hist_lin, is_linear=True)
                else:
                    self.publish_for_plotjuggler(real_hist_ang, is_linear=False)
                    
                
                
                
                stop = time.time()
                if (stop-start)*1000 >= 50:
                    print('execution took too long: {:.2f} ms'.format((stop-start)*1000))  
                
                
                
                
                #print('Time between samples: {}'.format(t_1-t_0))
                
                
                
                #print("Time diff real, simu: {}".format(self.imu_history[0][-1] * 1e-9 - t_1))
                #print("Linear accel real: {}".format(self.imu_history[0][0:3]))
                #print("Linear accel simu: {}\n".format(sim_accel))
                
                
                
                # optinally print position and attitude
                #print('Position: {}'.format(self.position))
                #print('Velocity: {}'.format(self.velocity))
                #print('Attitude: {}'.format(self.attitude))
                #print('Attitude: {}\n'.format(quaternion_to_euler_numpy(self.attitude)))
                
        elif self.offboard_setpoint_counter == 10:
                self.engage_offboard_mode()
                self.arm()        
        
        
          
            
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
        print(traceback.format_exc())
        print(e)