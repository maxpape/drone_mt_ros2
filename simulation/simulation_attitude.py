import rclpy
import numpy as np
import scipy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
from casadi import SX, vertcat, Function, sqrt, norm_2, dot, cross
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, ActuatorMotors, VehicleOdometry
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from drone_model_attitude import export_drone_ode_model


N_horizon = 20
Tf = 1
nx = 7
nu = 4
Tmax = 3
Tmin = 0




# parameters for ACAODS MPC
m = 1.5
g = -9.81
jxx = 0.029125
jyy = 0.029125
jzz = 0.055225
d_x0 = 0.107
d_x1 = 0.107 
d_x2 = 0.107 
d_x3 = 0.107 
d_y0 = 0.0935
d_y1 = 0.0935
d_y2 = 0.0935
d_y3 = 0.0935
c_tau = 0.000806428




current_state = np.asarray([1,0,0,0,0,0,0])
setpoint = np.asarray([1,0,0,0,0,0,0])



params = np.asarray([m,
                    g,
                    jxx,
                    jyy,
                    jzz,
                    d_x0,
                    d_x1,
                    d_x2, 
                    d_x3, 
                    d_y0,
                    d_y1,
                    d_y2,
                    d_y3,
                    c_tau])
        
        
        
parameters = np.concatenate((params, setpoint), axis=None)

def q_to_eu(q):
    quat = np.zeros(4)
    quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]
    
    rotation = R.from_quat(quat)
    
    return rotation.as_euler('xyz', degrees=True)
def q_to_eulers(array):
    
    res = list()
    
    for q in array:
        res.append(q_to_eu(q))
    return np.asarray(res)
        

def plot_drone_euler(N, Tf, simX, U):
    # Assuming 'input', 'x', and 'time' are your numpy arrays
    #time = np.linspace(0, N*Tf, N+1)
    time = np.linspace(0, N*0.05, num=N+1)

    N = len(time)

    # Determine the number of subplots
    num_subplots = U.shape[1] + simX.shape[1]-1

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots*3))
    U[-1,:] = 0

    # Plot the columns of 'U' array
    for i in range(U.shape[1]):
        
        
        axs[i].plot(time, U[:, i])
        axs[i].set_ylabel(f'U column {i+1}')

    # Plot the columns of 'simX' array
    for i in range(simX.shape[1]-1):
        if i < 3:
            axs[U.shape[1] + i].plot(time, q_to_eulers(simX[:, 0:4])[:,i])
            axs[U.shape[1] + i].set_ylabel(f'simX column {i+1}')
        else:
            axs[U.shape[1] + i].plot(time, simX[:, i+1])
            axs[U.shape[1] + i].set_ylabel(f'simX column {i+2}')
        
    

    # Set the x-label for the last subplot
    axs[-1].set_xlabel('Time')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def plot_drone(N, Tf, simX, U):
    # Assuming 'input', 'x', and 'time' are your numpy arrays
    #time = np.linspace(0, N*Tf, N+1)
    time = np.linspace(0, N*0.05, num=N+1)

    N = len(time)

    # Determine the number of subplots
    num_subplots = U.shape[1] + simX.shape[1]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots*3))
    U[-1,:] = 0

    # Plot the columns of 'U' array
    for i in range(U.shape[1]):
        
        
        axs[i].plot(time, U[:, i])
        axs[i].set_ylabel(f'U column {i+1}')

    # Plot the columns of 'simX' array
    for i in range(simX.shape[1]):

        axs[U.shape[1] + i].plot(time, simX[:, i])
        axs[U.shape[1] + i].set_ylabel(f'simX column {i+1}')
        
    

    # Set the x-label for the last subplot
    axs[-1].set_xlabel('Time')

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
        quat = SX.sym('quat', 4)
        quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]
        
        rotation = sc.Rotation.from_quat(quat)
        
        
        return rotation.as_euler('xyz')



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

def quaternion_error(q_ref, q):
    q_error = multiply_quaternions(q_ref, quaternion_inverse(q))
    
    return q_error

def error_function(y_ref, x):
    """Error function for MPC
        difference of position, velocity and angular velocity from reference
        use sub-function for calculating quaternion error
    """
    
    q_ref = y_ref[0:4]
    omega_ref = y_ref[4:7]
    
    q_err = quaternion_error(q_ref, x[0:4])
    omega_err = omega_ref - x[4:7]
    
    return vertcat(q_err, omega_err)

def setup(x0, N_horizon, Tf, RTI=False):
    ocp = AcadosOcp()
        
    # set model
    model = export_drone_ode_model()
    ocp.model = model
    

    ocp.dims.N = N_horizon
    ocp.dims.nh = 2
    ocp.dims.nh_e = 2
    
    ocp.parameter_values = parameters
    
    
    # define weighing matrices
    Q_mat = np.eye(7)*0.01
    Q_mat[0,0] = 1
    Q_mat[1,1] = 1
    Q_mat[2,2] = 1
    Q_mat[3,3] = 1
 
    
    
    R_mat = np.eye(4)*1
    
    Q_mat_final = np.eye(7)*0.02
    Q_mat_final[0,0] = 2
    Q_mat_final[1,1] = 2
    Q_mat_final[2,2] = 2
    Q_mat_final[3,3] = 2
     

    

    
    
    
    # set cost module
    x = ocp.model.x
    u = ocp.model.u
            
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    x_in = SX.sym('x', 7)
    
    q_ref = euler_to_quaternion(np.array([0,0,15]))
    y_ref = np.concatenate((q_ref, np.zeros(3)), axis=None)
    #error = Function('error', [y_ref, x_in], [vertcat(quaternion_error(y_ref[0:4], x_in[0:4]), (y_ref[4:] - x_in[4:]))])
    
    
    
    ocp.model.cost_expr_ext_cost = error_function(ocp.model.p[14:], ocp.model.x[0:7]).T @ Q_mat @ error_function(ocp.model.p[14:], ocp.model.x[0:7]) + u.T @ R_mat @ u
    ocp.model.cost_expr_ext_cost_e = error_function(ocp.model.p[14:], ocp.model.x[0:7]).T @ Q_mat_final @ error_function(ocp.model.p[14:], ocp.model.x[0:7])
    
    
    # set constraints
          
    ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
    ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    
    q_min = -0.2
    q_max = 0.2       
    ocp.constraints.lbx = np.array([q_min, q_min])
    ocp.constraints.ubx = np.array([q_max, q_max])
    ocp.constraints.idxbx = np.array([1, 2])
    
    
    # set initial state
    ocp.constraints.x0 = current_state
    angle_max = 30       
    
    ### constrain q to have norm = 1
    #q = SX.sym('q', 4)
    q = model.x[0:4]
    ##
    f_norm = Function('f_norm', [q], [sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)])
    f_roll = Function('f_roll', [q], [quaternion_to_euler(q)[0]])
    f_pitch = Function('f_pitch', [q], [quaternion_to_euler(q)[1]])
    ##
    ##
    ### constrain maximum angle of quadrotor
    ##max_angle = 30 * np.pi / 180
    ##
    ocp.model.con_h_expr = vertcat(f_norm(q), f_roll(q), f_pitch(q))
    ocp.constraints.lh = np.array([0.9, -angle_max, -angle_max]) # Lower bounds
    ocp.constraints.uh = np.array([1.1, +angle_max, +angle_max])  # Upper bounds
    
    ### copy for terminal shooting node
    ocp.model.con_h_expr_e = vertcat(f_norm(q), f_roll(q), f_pitch(q))
    ocp.constraints.lh_e = np.array([0.9, -angle_max, -angle_max])
    ocp.constraints.uh_e = np.array([1.1, +angle_max, +angle_max])
    #

    
    # set prediction horizon
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.tf = Tf
    
    # set solver options
    #ocp.solver_options.integrator_type = 'ERK'
    #ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    #ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.ext_cost_num_hess = True
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    
    # create ACADOS solver
    solver_json = 'acados_ocp_' + model.name + '.json'

    
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator


def main(use_RTI=False):


    ocp_solver, integrator = setup(current_state, N_horizon, Tf, use_RTI)

    

    Nsim = 200
    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim+1, nu))


    
    simX[0,:] = current_state

    
    t = np.zeros((Nsim))
    
    
    

    # closed loop
    for i in range(Nsim):

       
        
        if i >= 100:
            q_ref = euler_to_quaternion(np.array([0,0,15]))
            y_ref = np.concatenate((q_ref, np.zeros(3)), axis=None)
            parameters = np.concatenate((params, y_ref), axis=None)
            for j in range(N_horizon):
                
                ocp_solver.set(j, "p", parameters)
            ocp_solver.set(N_horizon, "p", parameters)
        
        
        
        #simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])
        
        t[i] = ocp_solver.get_stats('time_tot')

        # simulate system
        #simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])
        
        
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=np.array([1,0,0,1]))
    

    # plot results
    
    for x in simX:
        
        print(np.linalg.norm(x[0:4]))
    plot_drone(Nsim, Tf, simX, simU)
    ocp_solver = None
    integrator = None


if __name__ == '__main__':
    #main(use_RTI=False)
    main(use_RTI=True)