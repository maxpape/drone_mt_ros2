import rclpy
import numpy as np
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
from drone_model_attitude_thrust import export_drone_ode_model


N_horizon = 40
Tf = 2
nx = 11
nu = 4
Tmax = 10
Tmin = 0


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
#c_tau = 0.000806428
c_tau = 0.005

x0 = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
setpoint = np.asarray([1, 0, 0, 0, 0, 0, 0])



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

        axs[i].plot(time, U[:, i])
        axs[i].set_ylabel(f"U column {i+1}")

    # Plot the columns of 'simX' array
    for i in range(simX.shape[1] - 1):
        if i < 3:
            axs[U.shape[1] + i].plot(time, q_to_eulers(simX[:, 0:4])[:, i])
            axs[U.shape[1] + i].set_ylabel(f"simX column {i+1}")
        else:
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



def error_function(y_ref, x):
    """Error function for MPC
    difference of position, velocity and angular velocity from reference
    use sub-function for calculating quaternion error
    """

    q_ref = y_ref[0:4]
    omega_ref = y_ref[4:7]

    q_err = quaternion_error(y_ref[0:4], x[0:4])
    omega_err = y_ref[4:7] - x[4:7]

    
    
    return vertcat(q_err, omega_err)




def setup(x0, N_horizon, Tf, RTI=False):
    ocp = AcadosOcp()

    # set model
    model = export_drone_ode_model()
    ocp.model = model

    ocp.dims.N = N_horizon
    

    ocp.parameter_values = parameters

    # define weighing matrices
    Q_mat = np.eye(6)*0.1
    Q_mat[0, 0] = 2
    Q_mat[1, 1] = 2
    Q_mat[2, 2] = 2
    
    

    R_mat = np.eye(4) 

    Q_mat_final = np.eye(6)*0.1
    Q_mat_final[0, 0] = 2
    Q_mat_final[1, 1] = 2
    Q_mat_final[2, 2] = 2
    #Q_mat_final[3, 3] = 2

    # set cost module
    x = ocp.model.x[0:7]
    u = ocp.model.u
    ref = ocp.model.p[14:]
    
    
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    
    ocp.model.cost_expr_ext_cost =  (error_function(ref, x).T @ Q_mat @ error_function(ref, x)) + u.T @ R_mat @ u
    ocp.model.cost_expr_ext_cost_e = (error_function(ref, x).T @ Q_mat_final @ error_function(ref, x))

    # set constraints

    ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
    ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    
    max_angle_q = 0.2
    ocp.constraints.lbx = np.array([-max_angle_q, -max_angle_q])
    ocp.constraints.ubx = np.array([+max_angle_q, +max_angle_q])
    ocp.constraints.idxbx = np.array([1,2])

    # set initial state
    ocp.constraints.x0 = x0
    


    # set prediction horizon
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.tf = Tf



    ocp.solver_options.qp_solver_warm_start = 2
    ocp.solver_options.levenberg_marquardt = 20.0
    # create ACADOS solver
    solver_json = "acados_ocp_" + model.name + ".json"

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file=solver_json)

    return acados_ocp_solver, acados_integrator


def main(use_RTI=False):

    ocp_solver, integrator = setup(x0, N_horizon, Tf, use_RTI)

    Nsim = 200
    simX = np.ndarray((Nsim + 1, nx))
    simU = np.ndarray((Nsim + 1, nu))

    simX[0, :] = x0

    

    # closed loop
    for i in range(Nsim):
        
        # set different setpoint for attitude
        if i == 10:
            q_ref = euler_to_quaternion(np.array([0, 0, 0]))
            
            
            y_ref = np.concatenate((q_ref, np.array([0, 0, 0])), axis=None)
            
            parameters = np.concatenate((params, y_ref), axis=None)
            for j in range(N_horizon):

                ocp_solver.set(j, "p", parameters)
            ocp_solver.set(N_horizon, "p", parameters)
            
        
       

        
       
        simU[i, :] = ocp_solver.solve_for_x0(x0_bar=simX[i, :], fail_on_nonzero_status=True)
        
       
        

        # simulate system
        simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

        
        # manual control input
        #simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=np.array([5, 0, 5,0]))

    # plot results
    plot_drone_euler(Nsim, Tf, simX, simU)
    ocp_solver = None
    integrator = None



if __name__ == "__main__":
    # main(use_RTI=False)
    main(use_RTI=True)
