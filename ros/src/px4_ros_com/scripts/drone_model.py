from acados_template import AcadosModel
from casadi import SX, vertcat, horzcat, sum1, inv, cross, mtimes, dot, norm_2
import spatial_casadi as sc


def quaternion_inverse_casadi(q):
    """Invert a quaternion given as a casadi expression

    Args:
        q (casadi SX): input quaternion

    Returns:
        casadi SX: inverted quaternion
    """

    return SX([1, -1, -1, -1]) * q / (norm_2(q) ** 2)


def quaternion_product_casadi(q, p):
    """Multiply two quaternions given as casadi espressions

    Args:
        q (casadi SX): input quaternion q
        p (casadi SX): input quaternion p

    Returns:
        casadi SX: output quaternion
    """
    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - dot(v1, v2)
    v = s1 * v2 + s2 * v1 + cross(v1, v2)
    return vertcat(s, v)


def quat_derivative_casadi(q, w):
    """Calculates the quaternion derivative

    Args:
        q (casadi SX): input quaternion
        w (casadi SX): angular velocity

    Returns:
        casadi SX: quaternion derivative
    """

    return quaternion_product_casadi(q, vertcat(SX(0), w)) / 2


def quat_rotation_casadi(v, q):
    """Rotates a vector v by the quaternion q

    Args:
        v (casadi SX): input vector
        q (casadi SX): input quaternion

    Returns:
        _type_: _description_
    """

    p = vertcat(SX(0), v)
    p_rotated = quaternion_product_casadi(
        quaternion_product_casadi(q, p), quaternion_inverse_casadi(q)
    )
    return p_rotated[1:4]


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
    v_ref = SX.sym("v_ref", 3)
    w_ref = SX.sym("w_ref", 3)

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
        v_ref,
        w_ref,
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
        horzcat(-d_x0, -d_x1, +d_x2, +d_x3),
        horzcat(-d_y0, +d_y1, +d_y2, -d_y3),
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
        quat_derivative_casadi(q_WB, omega_B),
        quat_rotation_casadi(vertcat(0, 0, sum1(thrust)), q_WB) / m + vertcat(0, 0, g),
        inv(J) @ ((P @ thrust - cross(omega_B, J @ omega_B))),
        (thrust_set - thrust) * 25,
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
