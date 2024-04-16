#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosModel
from casadi import SX, vertcat, horzcat, sum1, inv, cross, mtimes, dot, norm_2, sqrt
import spatial_casadi as sc
import numpy as np


def sx_quat_inverse(q):

    return SX([1, -1, -1, -1]) * q / (norm_2(q) ** 2)


def sx_quat_multiply(q, p):

    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - dot(v1, v2)
    v = s1 * v2 + s2 * v1 + cross(v1, v2)
    return vertcat(s, v)


def quat_derivative(q, w):

    return sx_quat_multiply(q, vertcat(SX(0), w) / 2)


def quat_rotation(v, q):

    p = vertcat(SX(0), v)
    p_rotated = sx_quat_multiply(sx_quat_multiply(q, p), sx_quat_inverse(q))
    return p_rotated[1:4]


def quat_rotation_old(q, v):
    rot_mat = sc.Rotation.from_quat(q).as_matrix()

    rotated_vec = mtimes(rot_mat, v)

    return rotated_vec


# Function to compute the quaternion product matrix for quaternion multiplication
def quaternion_product_matrix(quat):

    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return vertcat(
        horzcat(w, -x, -y, -z),
        horzcat(x, w, -z, y),
        horzcat(y, z, w, -x),
        horzcat(z, -y, x, w),
    )


def export_drone_ode_model() -> AcadosModel:

    model_name = "drone_ode"

    # Define parameters
    m = SX.sym("m")  # Mass of the quadrotor
    g = SX.sym("g")  # Acceleration due to gravity
    jxx = SX.sym("jxx")
    jyy = SX.sym("jyy")
    jzz = SX.sym("jzz")

    d_x0 = SX.sym("d_x0")
    d_x1 = SX.sym("d_x1")
    d_x2 = SX.sym("d_x2")
    d_x3 = SX.sym("d_x3")
    d_y0 = SX.sym("d_y0")
    d_y1 = SX.sym("d_y1")
    d_y2 = SX.sym("d_y2")
    d_y3 = SX.sym("d_y3")
    c_tau = SX.sym("c_tau")

    q_ref = SX.sym("q_ref", 4)
    w_ref = SX.sym("w_ref", 3)

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
        q_ref,
        w_ref,
    )


    J = vertcat(
        horzcat(jxx, 0, 0), horzcat(0, jyy, 0), horzcat(0, 0, jzz)
    )  # Inertia matrix
    
    P = vertcat(
        horzcat(-d_x0, -d_x1, d_x2, d_x3),
        horzcat(d_y0, -d_y1, -d_y2, d_y3),
        horzcat(-c_tau, c_tau, -c_tau, c_tau),
    )

    # Define state variables
    q_WB = SX.sym("q_WB", 4)  # Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    omega_B = SX.sym("omega_B", 3)  # Angular velocity of the quadrotor in body frame
    T = SX.sym("T", 4)  # thrust force of the motors
    x = vertcat(q_WB, omega_B, T)

    # Define control inputs
    T_set = SX.sym("T_set", 4)  # Thrust produced by the rotors

    # xdot
    q_WB_dot = SX.sym("q_WB_dot", 4)  # derivative of Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    omega_B_dot = SX.sym("omega_B_dot", 3)  # derivative of Angular velocity of the quadrotor in body frame
    T_dot = SX.sym("T_dot", 4)  # derivative of thrust force motors
    xdot = vertcat(q_WB_dot, omega_B_dot, T_dot)

    f_expl = vertcat(
        0.5 * mtimes(quaternion_product_matrix(q_WB), vertcat(0, omega_B)),
        mtimes(inv(J), ((mtimes(P, T) - cross(omega_B, mtimes(J, omega_B))))),
        (T_set - T) * 10,
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = T_set
    model.p = params
    model.name = model_name

    return model
