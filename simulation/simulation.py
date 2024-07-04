import rclpy
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
from casadi import SX, vertcat, horzcat, Function, sqrt, norm_2, dot, cross, mtimes, atan2, if_else
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    ActuatorMotors,
    VehicleOdometry,
)
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from drone_model import export_drone_ode_model


N_horizon = 100
Tf = 5
nx = 17
nu = 4
Tmax = 10.0
Tmin = 0.5
vmax = 4.0
angular_vmax = 1.0


# parameters for ACAODS MPC
m = 1.5
g = -9.81
jxx = 0.0029125
jyy = 0.0029125
jzz = 0.0055225
d_x0 = 0.107
d_x1 = 0.107
d_x2 = 0.107
d_x3 = 0.107
d_y0 = 0.0935
d_y1 = 0.0935
d_y2 = 0.0935
d_y3 = 0.0935
#
#c_tau = 0.000806428
c_tau = 0.005

hover_thrust = -g*m/4


x0 = np.asarray([0,0,0,1, 0, 0, 0, 0, 0, 0, 0, 0, 0, hover_thrust, hover_thrust, hover_thrust, hover_thrust])
setpoint = np.asarray([0, 0, 0, 1, 0, 0, 0])



# fixed parameters
params = np.asarray(
    [m, g, jxx, jyy, jzz, d_x0, d_x1, d_x2, d_x3, d_y0, d_y1, d_y2, d_y3, c_tau]
)

# parameters with setpoint
parameters = np.concatenate((params, setpoint), axis=None)


def q_to_eu(q):
    quat = np.zeros(4)
    quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]

    rotation = R.from_quat(quat)

    return rotation.as_euler("xyz", degrees=True)


def q_to_eulers(array):

    res = list()

    for q in array:
        res.append(q_to_eu(q))
    return np.asarray(res)


def plot_drone_euler(N, Tf, simX, U):
    # Assuming 'input', 'x', and 'time' are your numpy arrays
    # time = np.linspace(0, N*Tf, N+1)
    time = np.linspace(0, N * 0.05, num=N + 1)

    N = len(time)

    # Determine the number of subplots
    num_subplots = U.shape[1] + simX.shape[1] - 1

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 3))
    U[-1, :] = 0

    # Plot the columns of 'U' array
    for i in range(U.shape[1]):

        axs[i].plot(time, (U[:, i]/10*1100))
        axs[i].set_ylabel(f"U column {i+1}")

    # Plot the columns of 'simX' array
    for i in range(simX.shape[1] - 1):
        if i <= 2:
            axs[U.shape[1] + i].plot(time, simX[:, i])
            axs[U.shape[1] + i].set_ylabel(f"simX column {i+1}")
        elif 2 < i < 6:
            axs[U.shape[1] + i].plot(time, q_to_eulers(simX[:, 3:7])[:, i-3])
            axs[U.shape[1] + i].set_ylabel(f"simX column {i+1}")
        elif 6 <= i:
            axs[U.shape[1] + i].plot(time, simX[:, i + 1])
            axs[U.shape[1] + i].set_ylabel(f"simX column {i+2}")

    # Set the x-label for the last subplot
    axs[-1].set_xlabel("Time")

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_drone(N, Tf, simX, U):
    # Assuming 'input', 'x', and 'time' are your numpy arrays
    # time = np.linspace(0, N*Tf, N+1)
    time = np.linspace(0, N * 0.05, num=N + 1)

    N = len(time)

    # Determine the number of subplots
    num_subplots = U.shape[1] + simX.shape[1]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 3))
    U[-1, :] = 0

    # Plot the columns of 'U' array
    for i in range(U.shape[1]):

        axs[i].plot(time, U[:, i])
        axs[i].set_ylabel(f"U column {i+1}")

    # Plot the columns of 'simX' array
    for i in range(simX.shape[1]):

        axs[U.shape[1] + i].plot(time, simX[:, i])
        axs[U.shape[1] + i].set_ylabel(f"simX column {i+1}")

    # Set the x-label for the last subplot
    axs[-1].set_xlabel("Time")

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def euler_to_quaternion(rpy):
    """
    Convert Euler angles to quaternion.ones

    Parameters:
    rpy : np.ndarray roll, pitch, yaw

    Returns:
    list
        Quaternion [w, x, y, z] representing the rotation.
    """
    roll, pitch, yaw = rpy
    # Create a rotation object from Euler angles
    r = R.from_euler("xyz", [roll, pitch, yaw], degrees=True)

    # Convert the rotation object to quaternion (scalar-last format)
    q = r.as_quat()

    return np.array([q[3], q[0], q[1], q[2]])


def quaternion_to_euler(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x-axis, pitch is rotation around y-axis,
    and yaw is rotation around z-axis.
    """
    quat = SX.sym("quat", 4)
    quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]

    rotation = sc.Rotation.from_quat(quat)

    return rotation.as_euler("xyz")




def multiply_quaternions(q, p):

    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - dot(v1, v2)
    v = s1 * v2 + s2 * v1 + cross(v1, v2)
    return vertcat(s, v)


def quaternion_inverse(q):

    return SX([1, -1, -1, -1]) * q / (norm_2(q) ** 2)


def quaternion_error(q_ref, q):

    q_error = multiply_quaternions(q_ref, quaternion_inverse(q))

    if_else(q_error[0] >= 0, SX([1, -1, -1, -1])*q_error, SX([1, 1, 1, 1])*q_error, True)
    

    return q_error[1:4]






def error_function(x, y_ref):
    """Error function for MPC
        difference of position, velocity and angular velocity from reference
        use sub-function for calculating quaternion error
    """
    p_ref = y_ref[0:3]
    q_ref = y_ref[3:7]
    
    
    p_err = p_ref - x[0:3]
    q_err = quaternion_error(q_ref, x[3:7])
    
    
    return vertcat(p_err, q_err)

def setup(x0, N_horizon, Tf, RTI=False):
    ocp = AcadosOcp()

    # set model
    model = export_drone_ode_model()
    ocp.model = model

    ocp.dims.N = N_horizon
    

    ocp.parameter_values = np.concatenate((parameters), axis=None)

    # define weighing matrices
    Q_p = np.diag([1,1,1000])
    Q_q = np.diag([1,1,3])*0.001
    Q_mat = scipy.linalg.block_diag(Q_p, Q_q)
   

    R_mat = np.eye(4)

    Q_p_final = np.diag([1,1,1000])
    Q_q_final = np.diag([1,1,3])*0.001
    Q_mat_final = scipy.linalg.block_diag(Q_p_final, Q_q_final)

    # set cost module
    x = ocp.model.x[0:7]
    u = ocp.model.u
    ref = ocp.model.p[14:21]
    
    
    
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    
    ocp.model.cost_expr_ext_cost =  error_function(ref, x).T @ Q_mat @ error_function(ref, x) + u.T @ R_mat @ u
    ocp.model.cost_expr_ext_cost_e = error_function(ref, x).T @ Q_mat_final @ error_function(ref, x)

    # set constraints
        
    
            
    ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
    ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    
    #ocp.constraints.lbx = np.array([-vmax, -vmax, -vmax, -angular_vmax, -angular_vmax, -angular_vmax, Tmin, Tmin, Tmin, Tmin])
    #ocp.constraints.ubx = np.array([+vmax, +vmax, +vmax, +angular_vmax, +angular_vmax, +angular_vmax, Tmax, Tmax, Tmax, Tmax])
    #ocp.constraints.idxbx = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
              

    # set initial state
    ocp.constraints.x0 = x0
    


    # set prediction horizon
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.tf = Tf


    ocp.solver_options.levenberg_marquardt = 10.0
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_OSQP'
    # create ACADOS solver
    solver_json = "acados_ocp_" + model.name + ".json"

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file=solver_json)

    return acados_ocp_solver, acados_integrator


def main(use_RTI=False):

    ocp_solver, integrator = setup(x0, N_horizon, Tf, use_RTI)

    Nsim = 10
    simX = np.ndarray((Nsim + 1, nx))
    simU = np.ndarray((Nsim + 1, nu))

    simX[0, :] = x0

    

    # closed loop
    for i in range(Nsim):
        
        
        simU[i, :] = np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])
        simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])
        
        
        
    #
    #
    v0 = simX[:,7:10][:-1]
    v1 = simX[:,7:10][1:]
    
    
    a = (v1-v0) *20
    print(a)   
        
    x_0 = simX[0,7:10]
    x_1 = simX[1,7:10]
    x_2 = simX[2,7:10]
    x_3 = simX[3,7:10]
    x_4 = simX[4,7:10]
    #
    #
    #x_0 = x_0  
    #x_1 = x_1 - 0.1
    #x_2 = x_2 - 0.2
    #x_3 = x_3 - 0.3
    #x_4 = x_4 - 0.4
    #print(a-0.05)
    
    print('x0: {}'.format(x_0))
    print('x1: {}'.format(x_1))
    print('x2: {}'.format(x_2))
    print('x3: {}'.format(x_3))
    print('x4: {}'.format(x_4))
        #
        
        
        
        #thrust = hover_thrust
        #
        ## manual control input
        #simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=np.ones(4)*hover_thrust)

    # plot results
    simx = np.zeros((Nsim+1, nx+1))
    simx[:, 0] = np.linspace(0, Tf/N_horizon*Nsim, Nsim+1)
    simx[:,1:nx+1] = simX
    
    simu = np.zeros((Nsim+1, nu+1))
    simu[:, 0] = np.linspace(0, Tf/N_horizon*Nsim, Nsim+1)
    simu[:,1:nu+1] = simU
    
    
    simx_pd = pd.DataFrame(simx)
    simu_pd = pd.DataFrame(simu)
    
    columns_x = ['t', 'px', 'py', 'pz', 'q_w', 'q_x', 'q_y', 'q_z', 'vx', 'vy', 'vz', 'w_x', 'w_y', 'w_z', 'motor_FR', 'motor_BR', 'motor_BL', 'motor_FL']
    columns_u = ['t', 'motor_FR_set', 'motor_BR_set', 'motor_BL_set', 'motor_FL_set']
    simx_pd.to_csv("simx_pos.csv", header=columns_x)
    simu_pd.to_csv('simu_pos.csv', header=columns_u) 
    plot_drone(Nsim, Tf, simX, simU)
    ocp_solver = None
    integrator = None



if __name__ == "__main__":
    # main(use_RTI=False)
    main(use_RTI=True)
