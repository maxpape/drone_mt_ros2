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
        w_ref,
    )

    # fill parameter matrices
    J = vertcat(
        horzcat(jxx, 0, 0), horzcat(0, jyy, 0), horzcat(0, 0, jzz)
    )  # Inertia matrix
    P = vertcat(
        horzcat(-d_x0, -d_x1, d_x2, d_x3),
        horzcat(d_y0, -d_y1, -d_y2, d_y3),
        horzcat(-c_tau, c_tau, -c_tau, c_tau),
    )

    # Define state variables

    omega_B = SX.sym("omega_B", 3)  # Angular velocity of the quadrotor in body frame
    T = SX.sym("T", 4)  # thrust force of the motors
    x = vertcat(omega_B, T)

    # Define control inputs
    T_set = SX.sym("T_set", 4)  # Thrust produced by the rotors

    # xdot
    omega_B_dot = SX.sym(
        "omega_B_dot", 3
    )  # derivative of Angular velocity of the quadrotor in body frame
    T_dot = SX.sym("T_dot", 4)  # derivative of thrust force motors
    xdot = vertcat(omega_B_dot, T_dot)

    f_expl = vertcat(
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
