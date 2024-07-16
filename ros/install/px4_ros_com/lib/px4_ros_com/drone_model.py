from acados_template import AcadosModel
from casadi import SX, vertcat, horzcat, sum1, inv, cross, mtimes, dot, norm_2
import spatial_casadi as sc
import functions










def export_drone_ode_model() -> AcadosModel:

    model_name = "drone_ode"

    # Define parameters
    m = SX.sym("m")  # Mass of the quadrotor
    g = SX.sym("g")  # Acceleration due to gravity
    jxx = SX.sym("jxx")  # diagonal components of inertia matrix
    jyy = SX.sym("jyy")
    jzz = SX.sym("jzz")

    d_x0 = SX.sym("d_x0")  # distances of motors from respective axis
    d_x1 = SX.sym("d_x1")
    d_x2 = SX.sym("d_x2")
    d_x3 = SX.sym("d_x3")
    d_y0 = SX.sym("d_y0")
    d_y1 = SX.sym("d_y1")
    d_y2 = SX.sym("d_y2")
    d_y3 = SX.sym("d_y3")
    c_tau = SX.sym("c_tau")  # rotor drag torque constant
    p_ref = SX.sym("p_ref", 3)  # reference variables for setpoint
    q_ref = SX.sym("q_ref", 4)
    lin_acc_offset = SX.sym("lin_acc_offset", 3)
    ang_acc_offset = SX.sym("ang_acc_offset", 3)
    
    

    # combine parameters to single vector
    params = vertcat(
        m,
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
        c_tau,
        p_ref,
        q_ref,
        lin_acc_offset,
        ang_acc_offset
    )

    # Define state variables
    p_WB = SX.sym("p_WB", 3)  # Position of the quadrotor (x, y, z)
    q_WB = SX.sym(
        "q_WB", 4
    )  # Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    v_WB = SX.sym("v_WB", 3)  # Linear velocity of the quadrotor
    omega_B = SX.sym("omega_B", 3)  # Angular velocity of the quadrotor in body frame
    thrust = SX.sym("T", 4)
    x = vertcat(p_WB, q_WB, v_WB, omega_B, thrust)

    # Define control inputs
    thrust_set = SX.sym("T_set", 4)  # Thrust produced by the rotors

    # Inertia matrix
    J = vertcat(horzcat(jxx, 0, 0),
                horzcat(0, jyy, 0),
                horzcat(0, 0, jzz))

    # thrust allocation matrix
    P = vertcat(
        horzcat(-d_x0, +d_x1, +d_x2, -d_x3),
        horzcat(-d_y0, +d_y1, -d_y2, +d_y3),
        horzcat(-c_tau, c_tau, -c_tau, c_tau),
    )

    # xdot
    p_WB_dot = SX.sym("p_WB_dot", 3)  # derivative of Position of the quadrotor (x, y, z)
    q_WB_dot = SX.sym("q_WB_dot", 4)  # derivative of Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    v_WB_dot = SX.sym("v_WB_dot", 3)  # derivative of Linear velocity of the quadrotor
    omega_B_dot = SX.sym("omega_B_dot", 3)  # derivative of Angular velocity of the quadrotor in body frame
    thrust_dot = SX.sym("T_dot", 4)   # derivative of thrust

    xdot = vertcat(p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, thrust_dot)

    f_expl = vertcat(
        v_WB,
        functions.quat_derivative_casadi(q_WB, omega_B),
        functions.quat_rotation_casadi(vertcat(0, 0, sum1(thrust)), q_WB) / m + vertcat(0, 0, g) + lin_acc_offset,
        inv(J) @ ((P @ thrust - cross(omega_B, J @ omega_B))) + ang_acc_offset,
        (thrust_set - thrust) * 125,
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = thrust_set
    model.p = params
    model.name = model_name

    return model
