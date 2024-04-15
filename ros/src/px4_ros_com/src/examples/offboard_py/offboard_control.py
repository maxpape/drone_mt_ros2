#!/usr/bin/env python3

import rclpy
import numpy as np
import scipy.linalg
import scipy.interpolate
from casadi import SX, vertcat, Function, sqrt, norm_2, dot, cross, atan2
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, ActuatorMotors, VehicleOdometry, ActuatorOutputs
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from drone_model import export_drone_ode_model


def euler_to_quaternion(rpy):
    """
    Convert Euler angles to quaternion.

    Parameters:
    rpy : np.ndarray roll, pitch, yaw

    Returns:
    list
        Quaternion [w, x, y, z] representing the rotation.
    """
    roll, pitch, yaw = rpy
    # Create a rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    
    # Convert the rotation object to quaternion (scalar-last format)
    q = r.as_quat()
    
    return np.array([q[3], q[0], q[1], q[2]] )



def quaternion_to_euler(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x-axis, pitch is rotation around y-axis,
        and yaw is rotation around z-axis.
        """
        quat = SX.sym("quat", 4)
        quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]
        rotation = sc.Rotation.from_quat(quat)
        
        
        return rotation.as_euler('xyz')

def q_to_eu(q):
    quat = np.zeros(4)
    quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]

    rotation = R.from_quat(quat)

    return rotation.as_euler("xyz", degrees=True)

def multiply_quaternions2(q1, q2):
    # Extract components of the first quaternion
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    
    # Extract components of the second quaternion
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    # Compute the product of the two quaternions
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # Return the result as a new quaternion
    return vertcat(w, x, y, z)


def multiply_quaternions(q, p):

    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - dot(v1, v2)
    v = s1 * v2 + s2 * v1 + cross(v1, v2)
    return vertcat(s, v)

def quaternion_inverse(q):

    return SX([1, -1, -1, -1]) * q / (norm_2(q)**2)

def quaternion_error(q, q_ref):
    q_error = multiply_quaternions(q_ref, quaternion_inverse(q))
    
    return q_error

def error_funciton_old(x, y_ref):
    """Error function for MPC
        difference of position, velocity and angular velocity from reference
        use sub-function for calculating quaternion error
    """
    p_ref = y_ref[0:3]
    q_ref = y_ref[3:7]
    
    
    p_err = x[0:3] - p_ref
    q_err = quaternion_error(x[3:7], q_ref)
    
    return q_err
def error_funciton(x, y_ref):
    """Error function for MPC
        difference of position, velocity and angular velocity from reference
        use sub-function for calculating quaternion error
    """
    p_ref = y_ref[0:3]
    q_ref = y_ref[3:7]
    
    
    p_err = x[0:3] - p_ref
    q_err = quaternion_error(x[3:7], q_ref)
    q_err = 2*atan2(norm_2(q_err[1:]), q_err[0])
    
    return vertcat(p_err, q_err)


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
        self.motor_command_publisher_pseudo = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors_pseudo', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_motor_subscriber = self.create_subscription(
            ActuatorOutputs, '/fmu/out/actuator_outputs', self.vehicle_motor_callback, qos_profile)



        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.ocp_solver = None
        
        #state variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.asarray([np.sqrt(2)/2,0,0,-np.sqrt(2)/2])
        self.angular_velocity = np.zeros(3)
        self.thrust = np.zeros(4)
        self.current_state = np.concatenate((self.position, self.attitude, self.velocity, self.angular_velocity, self.thrust), axis=None)
        
        #setpoint variables
        self.position_setpoint = np.zeros(3)
        self.velocity_setpoint = np.zeros(3)
        self.attitude_setpoint = np.asarray([np.sqrt(2)/2,0,0,-np.sqrt(2)/2])
        self.angular_velocity_setpoint = np.zeros(3)
        #self.thrust_setpoint = np.zeros(4)
        self.setpoint = np.concatenate((self.position_setpoint, self.attitude_setpoint, self.velocity_setpoint, self.angular_velocity_setpoint), axis=None)
        

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
        
        
        # Declare a parameter with a descriptor for dynamic reconfiguration
        angle_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,  # Specify the type as double
            description='Desired pangle',     # Description of the parameter
            additional_constraints='Range: -180 to 180',
            floating_point_range=[FloatingPointRange(
                from_value=-180.0,  # Minimum value
                to_value=180.0,     # Maximum value
                step=1.0         # Step size (optional)
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
        
        
        
        
        
        self.N_horizon = 40
        self.Tf = 2
        self.nx = 17
        self.nu = 4
        self.Tmax = 5
        self.Tmin = 0.1
        self.max_motor_rpm = 1100
        
        # parameters for ACAODS MPC
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
        #self.c_tau = 0.00806428
        
        
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
        
        
        
        self.parameters = np.concatenate((self.params, self.setpoint), axis=None)
        
    
    

        
    def setup_mpc(self):
        
        ocp = AcadosOcp()
        
        # set model
        
        model = export_drone_ode_model()
        ocp.model = model
        

        ocp.dims.N = self.N_horizon
        
        ocp.parameter_values = self.parameters
        
        
        # define weighing matrices
        Q_mat = np.eye(4)
        #Q_mat[0,0] = 0.5
        #Q_mat[1,1] = 0.5
        #Q_mat[2,2] = 0.5
        #Q_mat[3,3] = 80
        #Q_mat[4,4] = 80
        #Q_mat[5,5] = 80
        
        R_mat = np.eye(4)*0.2
        
        Q_mat_final = np.eye(4)
        #Q_mat_final[0,0] = 2
        #Q_mat_final[1,1] = 2
        #Q_mat_final[2,2] = 4
        #Q_mat_final[3,3] = 200
        #Q_mat_final[4,4] = 200
        #Q_mat_final[5,5] = 200       

        
        
        
        
        
        
        # set cost module
        x = ocp.model.x[0:7]
        u = ocp.model.u
        ref = ocp.model.p[14:21]
        
        
        
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        
        
        ocp.model.cost_expr_ext_cost = error_funciton(x, ref).T @ Q_mat @ error_funciton(x,ref) + u.T @ R_mat @ u
        ocp.model.cost_expr_ext_cost_e = error_funciton(x, ref).T @ Q_mat_final @ error_funciton(x, ref)
        
        
        # set constraints
        Tmin = self.Tmin
        Tmax = self.Tmax        
        ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
        ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    
        #vmax_angle = 1.6
        #ocp.constraints.lbx = np.array([-vmax_angle, -vmax_angle, -vmax_angle])
        #ocp.constraints.ubx = np.array([+vmax_angle, +vmax_angle, +vmax_angle])
        #ocp.constraints.idxbx = np.array([10, 11, 12])
              
        
        
        
        # set initial state
        current_state = np.concatenate((self.position, self.attitude, self.velocity, self.angular_velocity, self.thrust), axis=None)
        ocp.constraints.x0 = current_state
                
        
        # constrain q to have norm = 1
        #q = SX.sym('q', 4)
        ##
        #f_norm = Function('f_norm', [q], [sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)])
        #f_roll = Function('f_roll', [q], [quaternion_to_euler(q)[0]])
        #f_pitch = Function('f_pitch', [q], [quaternion_to_euler(q)[1]])
        ##
        ##
        ### constrain maximum angle of quadrotor
        #max_angle = 30 * np.pi / 180
        ##
        #ocp.model.con_h_expr = f_norm(model.x[3:7])
        #ocp.constraints.lh = np.array([0.99]) # Lower bounds
        #ocp.constraints.uh = np.array([1.01])  # Upper bounds
        #
        ### copy for terminal shooting node
        #ocp.constraints.uh_e = ocp.constraints.uh
        #ocp.constraints.lh_e = ocp.constraints.lh
        #ocp.model.con_h_expr_e = ocp.model.con_h_expr
        
        
        
        # set prediction horizon
        ocp.solver_options.qp_solver_cond_N = self.N_horizon
        ocp.solver_options.tf = self.Tf
        
        # set solver options
        #ocp.solver_options.integrator_type = 'IRK'
        #ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        #ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        #ocp.solver_options.ext_cost_num_hess = True
        
        # create ACADOS solver
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
        
        #self.attitude = self.NED_to_ENU(vehicle_odometry.q)
        self.attitude = self.NED_to_ENU(np.array([1,0,0,0]))
        self.angular_velocity = self.NED_to_ENU(vehicle_odometry.angular_velocity)
              
    
    
    def vehicle_motor_callback(self, motor_output):
        
        
        m0 = motor_output.output[0]
        m1 = motor_output.output[1]
        m2 = motor_output.output[2]
        m3 = motor_output.output[3]
        
        thrust = np.zeros(4)
        
        thrust[0] = m0 / self.max_motor_rpm
        thrust[1] = m1 / self.max_motor_rpm
        thrust[2] = m2 / self.max_motor_rpm
        thrust[3] = m3 / self.max_motor_rpm
        
        
        self.thrust = thrust*10
        
        
    
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
             
        
        parameters = np.concatenate((self.params, self.setpoint), axis=None)
        
        
        for j in range(self.N_horizon):
            
            
            self.ocp_solver.set(j, "p", parameters)
        
        self.ocp_solver.set(self.N_horizon, "p", parameters)
        
    def map_logarithmic(self, input_value):
        # Ensure the input is within the expected range
        #if input_value < 0 or input_value > 10:
         #   raise ValueError("Input value must be between 0 and 10")
        
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
        
        #params = self.get_parameters(
        #    ['motor_speed_0', 'motor_speed_1', 'motor_speed_2', 'motor_speed_3'])
        #
        #control = np.asarray([p.value for p in params])
        
        msg.control[0] = control[0]  # Motor 1
        msg.control[1] = control[2]  # Motor 2
        msg.control[2] = control[3]  # Motor 3
        msg.control[3] = control[1]  # Motor 4
        
        self.motor_command_publisher.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % control)
        
    def publish_motor_command_pseudo(self, control):
        """Publish the motor command setpoint."""
        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        msg.control[0] = control[0]  # Motor 1
        msg.control[1] = control[1]  # Motor 2
        msg.control[2] = control[2]  # Motor 3
        msg.control[3] = control[3]  # Motor 4
        
        self.motor_command_publisher_pseudo.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % control)
    
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
        #if True:
            # if in offboard mode: get setpoint from parameters, get optimal U, publish motor command
            params = self.get_parameters(
            ['position_x', 'position_y', 'position_z'])
            position_setpoint = np.asarray([p.value for p in params])
            
            params = self.get_parameters(
            ['roll', 'pitch', 'yaw'])
            rpy = np.asarray([p.value for p in params])
            attitude_setpoint = euler_to_quaternion(rpy)
            
            
            
            self.setpoint = np.concatenate((position_setpoint, attitude_setpoint, self.velocity_setpoint, self.angular_velocity_setpoint), axis=None)
            current_state = np.concatenate((self.position, self.attitude, self.velocity, self.angular_velocity, self.thrust), axis=None)
            
            
            self.set_mpc_target_pos()
           
            #self.ocp_solver.set(0, "lbx", current_state)
            #self.ocp_solver.set(0, "ubx", current_state) 
            
            #print("current state: %f", self.current_state)
            #print("current setpoint: %f", self.setpoint)
            #status = self.ocp_solver.solve()
            #U = self.ocp_solver.get(0, 'u')
            
            U = self.ocp_solver.solve_for_x0(x0_bar = current_state, fail_on_nonzero_status=False)
            
            q = self.ocp_solver.get(40, 'x')[3:7]
            cost = self.ocp_solver.get_cost()
            #print(cost)
            #print(q_to_eu(self.attitude))
            #self.current_state[13:] = U
            
            
            
            
            command = np.asarray([self.map_logarithmic(u)*0.8 for u in U])
            #print(self.parameters[14:])
            #print(self.ocp_solver.get_cost())
            #print(U)
            #print(quaternion_error(self.attitude_setpoint, self.attitude))
            #print(rpy)
            
            #print(self.setpoint)
            #print(current_state)
            print(command)
            #
            #print(self.attitude)
            #print(attitude_setpoint)
            #print(command)
            self.publish_motor_command(command)
            #self.publish_motor_command_pseudo(command)
            
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
