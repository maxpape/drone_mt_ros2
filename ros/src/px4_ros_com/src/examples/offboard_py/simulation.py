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
from drone_model_simulation import export_drone_ode_model

speed = np.zeros(4)
N_horizon = 100
Tf = 2
nx = 17
nu = 4
Tmax = 10
Tmin = 0


current_state = np.asarray([0,0,0,np.sqrt(2)/2,0,0,-np.sqrt(2)/2,0,0,0,0,0,0,0,0,0,0])
setpoint = np.asarray([0,0,0,np.sqrt(2)/2,0,0,-np.sqrt(2)/2,0,0,0,0,0,0,0,0,0,0])

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





yref = np.zeros((nx, ))
yref[2] = 5
yref[0:3] = setpoint[0:3]
yref[3] = np.sqrt(2)/2
yref[6] = -np.sqrt(2)/2

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
        
        
        
parameters = np.concatenate((params, yref), axis=None)

def plot_drone(N, Tf, simX, U):
    # Assuming 'input', 'x', and 'time' are your numpy arrays
    #time = np.linspace(0, N*Tf, N+1)
    time = np.linspace(0, Tf, num=N+1)

    N = len(time)

    # Determine the number of subplots
    num_subplots = U.shape[1] + simX.shape[1]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots*3))
    

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
        rotation = sc.Rotation.from_quat(q)
        
        
        return rotation.as_euler('zyx')

def q_to_eu(q):
    rotation = R.from_quat(q)
    
    return rotation.as_euler('zyx', degrees=True)

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

def error_funciton(x, y_ref):
    """Error function for MPC
        difference of position, velocity and angular velocity from reference
        use sub-function for calculating quaternion error
    """
    p_ref = y_ref[0:3]
    q_ref = y_ref[3:7]
    v_ref = y_ref[7:10]
    omega_ref = y_ref[10:13]
    T_ref = y_ref[13:17]

    p_err = x[0:3] - p_ref
    q_err = quaternion_error(x[3:7], q_ref)
    v_err = x[7:10] - v_ref
    omega_err = x[10:13] - omega_ref
    T_err = x[13:17] - T_ref
    
    

    return vertcat(p_err, q_err, v_err, omega_err, T_err)


def setup(x0, N_horizon, Tf, RTI=False):
    ocp = AcadosOcp()
        
    # set model
    model = export_drone_ode_model()
    ocp.model = model
    

    ocp.dims.N = N_horizon
    
    ocp.parameter_values = parameters
    
    
    # define weighing matrices
    Q_mat = np.zeros((17,17))
    Q_mat[0,0] = 1
    Q_mat[1,1] = 1
    Q_mat[2,2] = 1
    
    R_mat = np.eye(4)
    
    Q_mat_final = np.eye(17)
    Q_mat_final[0,0] = 2
    Q_mat_final[1,1] = 2
    Q_mat_final[2,2] = 2
    Q_mat_final[13,13] = 0
    Q_mat_final[14,14] = 0
    Q_mat_final[15,15] = 0
    Q_mat_final[16,16] = 0        

    

    
    
    
    # set cost module
    x = ocp.model.x
    u = ocp.model.u
            
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    
    ocp.model.cost_expr_ext_cost = error_funciton(x, ocp.model.p[14:]).T @ Q_mat @ error_funciton(x, ocp.model.p[14:]) + u.T @ R_mat @ u
    ocp.model.cost_expr_ext_cost_e = error_funciton(x, ocp.model.p[14:]).T @ Q_mat_final @ error_funciton(x, ocp.model.p[14:])
    
    
    # set constraints
          
    ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
    ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    #vmax_angle = 1.6
    #ocp.constraints.lbx = np.array([-vmax_angle, -vmax_angle, -vmax_angle])
    #ocp.constraints.ubx = np.array([+vmax_angle, +vmax_angle, +vmax_angle])
    #ocp.constraints.idxbx = np.array([10, 11, 12])
            
    
    
    
    # set initial state
    ocp.constraints.x0 = current_state
            
    
    ## constrain q to have norm = 1
    q = SX.sym('q', 4)
    #
    f_norm = Function('f_norm', [q], [sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)])
    f_roll = Function('f_roll', [q], [quaternion_to_euler(q)[0]])
    f_pitch = Function('f_pitch', [q], [quaternion_to_euler(q)[1]])
    #
    #
    ## constrain maximum angle of quadrotor
    max_angle = 30 * np.pi / 180
    #
    ocp.model.con_h_expr = f_norm(model.x[3:7])
    ocp.constraints.lh = np.array([0.99]) # Lower bounds
    ocp.constraints.uh = np.array([1.01])  # Upper bounds
    
    ## copy for terminal shooting node
    ocp.constraints.uh_e = ocp.constraints.uh
    ocp.constraints.lh_e = ocp.constraints.lh
    ocp.model.con_h_expr_e = ocp.model.con_h_expr
    
    
    
    # set prediction horizon
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.tf = Tf
    
    # set solver options
    #ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    #ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.ext_cost_num_hess = True
    
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
    
    t_preparation = np.zeros((Nsim))
    t_feedback = np.zeros((Nsim))

    

    # closed loop
    for i in range(Nsim):

        
         
         
        
         
        
        #if i == 1 :
        #    
        #    
        #    
        #    for j in range(N_horizon):
        #        
        #        ocp_solver.set(j, "p", parameters)
        #        
        #    
        #    
        #    
        #    ocp_solver.set(N_horizon, "p", parameters)  
         
           
        #if i == 50 :
        #    
        #    lbx = np.asarray([-6, -6, -6, -10, -10, -10])
        #    ubx = np.asarray([6, 6, 6, 10, 10, 10])
        #    
        #    for j in range(N_horizon):
        #        yref = np.array([108, 108, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #        ocp_solver.set(j, "yref", yref)
        #        
        #    
        #    ocp_solver.set(0, "lbx", simX[i, :])
        #    ocp_solver.set(0, "ubx", simX[i, :])   
        #    for j in range(1, N_horizon):
        #        ocp_solver.set(j, "lbx", lbx[0:6])
        #        ocp_solver.set(j, "ubx", ubx[0:6])
        #    yref_N = np.array([108, 108, 108, 0, 0, 0, 0, 0, 0])
        #    ocp_solver.set(N_horizon, "yref", yref_N)
        #
        ##ocp_solver.set(0, "lbx", simX[i, :])
        ##ocp_solver.set(0, "ubx", simX[i, :])
        # solve ocp and get next control input
        #ocp_solver.set(0, "lbx", simX[i, :])
        #ocp_solver.set(0, "ubx", simX[i, :]) 
        #status = ocp_solver.solve()
        simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])
        #simU[i,:] = ocp_solver.get(0, "u")
        #simU[i,:] = np.ones(4)*8
        t[i] = ocp_solver.get_stats('time_tot')

        # simulate system
        
        #
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])
        #simX[i+1, :] = integrator.simulate(x=simX[i, :], u=np.zeros(4)*8)
    

    # plot results
    

    plot_drone(Nsim, Tf, simX, simU)
    ocp_solver = None
    integrator = None


if __name__ == '__main__':
    #main(use_RTI=False)
    main(use_RTI=True)