import rclpy
import numpy as np
import scipy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
from casadi import SX, vertcat, horzcat, Function, sqrt, norm_2, dot, cross, mtimes
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
from drone_model_ang_rate_thrust import export_drone_ode_model


N_horizon = 20
Tf = 1
nx = 7
nu = 4
Tmax = 10
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


x0 = np.asarray([0, 0, 0, 0, 0, 0, 0])
setpoint = np.asarray([0, 0, 0])


params = np.asarray(
    [m, g, jxx, jyy, jzz, d_x0, d_x1, d_x2, d_x3, d_y0, d_y1, d_y2, d_y3, c_tau]
)


parameters = np.concatenate((params, setpoint), axis=None)


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


def error_function(y_ref, x):
    """Error function for MPC
    difference of position, velocity and angular velocity from reference
    use sub-function for calculating quaternion error
    """

    omega_err = y_ref - x

    return omega_err


def setup(x0, N_horizon, Tf, RTI=False):
    ocp = AcadosOcp()

    # set model
    model = export_drone_ode_model()
    ocp.model = model

    ocp.dims.N = N_horizon

    ocp.parameter_values = parameters

    # define weighing matrices
    Q_mat = np.eye(3)

    R_mat = np.eye(4)*0.01

    Q_mat_final = np.eye(3)

    # set cost module
    x = ocp.model.x[0:3]
    u = ocp.model.u
    ref = ocp.model.p[14:]

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost = (
        error_function(ref, x).T @ Q_mat @ error_function(ref, x) + u.T @ R_mat @ u
    )
    ocp.model.cost_expr_ext_cost_e = (
        error_function(ref, x).T @ Q_mat_final @ error_function(ref, x)
    )

    # set constraints

    ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
    ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
    ocp.constraints.idxbu = np.array([0,1,2,3])

    # set initial state
    ocp.constraints.x0 = x0

    # set prediction horizon
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.tf = Tf

    # set solver options
    # ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    # ocp.solver_options.ext_cost_num_hess = True
    # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

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

        if i == 100:
           
           
           parameters = np.concatenate((params, np.array([10, 0, 0])), axis=None)
           for j in range(N_horizon):
               ocp_solver.set(j, "p", parameters)
           ocp_solver.set(N_horizon, "p", parameters)
        
        

        # alternative method of getting U
        # ocp_solver.set(0, 'lbx', simX[i, :])
        # ocp_solver.set(0, 'ubx', simX[i, :])
        # status = ocp_solver.solve()
        # simU[i, :] = ocp_solver.get(0, "u")

        # get optimal U
        simU[i, :] = ocp_solver.solve_for_x0(x0_bar=simX[i, :])

        # simulate system with optimal U
        simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])


        # print cost
        print(ocp_solver.get_cost())
        # manual input for verification
        # simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=np.array([0, 0, 0,0]))

    # plot results
    plot_drone(Nsim, Tf, simX, simU)
    ocp_solver = None
    integrator = None


if __name__ == "__main__":
    # main(use_RTI=False)
    main(use_RTI=True)
