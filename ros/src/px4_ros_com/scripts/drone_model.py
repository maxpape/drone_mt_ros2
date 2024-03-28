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
from casadi import SX, vertcat, sum1, inv, cross 

def export_drone_ode_model() -> AcadosModel:

    model_name = 'drone_ode'

    

    # Define state variables
    p_WB = SX.sym('p_WB', 3)  # Position of the quadrotor (x, y, z)
    q_WB = SX.sym('q_WB', 4)  # Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    v_WB = SX.sym('v_WB', 3)  # Linear velocity of the quadrotor
    omega_B = SX.sym('omega_B', 3)  # Angular velocity of the quadrotor in body frame

    x = vertcat(p_WB, q_WB, v_WB, omega_B)

    # Define control inputs
    thrust = SX.sym('T', 4)  # Thrust produced by the rotors
    
    # Define parameters
    m = SX.sym('m')  # Mass of the quadrotor
    g = SX.sym('g')  # Acceleration due to gravity
    J = SX.sym('J', 3, 3)  # Inertia matrix
    P = SX.sym('P', (3,4))
    
    params = vertcat(m, g, J, P)

    
    
    # xdot
    p_WB_dot = SX.sym('p_WB_dot', 3)        # derivative of Position of the quadrotor (x, y, z)
    q_WB_dot = SX.sym('q_WB_dot', 4)        # derivative of Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    v_WB_dot = SX.sym('v_WB_dot', 3)        # derivative of Linear velocity of the quadrotor
    omega_B_dot = SX.sym('omega_B_dot', 3)  # derivative of Angular velocity of the quadrotor in body frame

    xdot = vertcat(p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot)


    # dynamics
    f_expl = vertcat(v_WB,
                     q_WB *  vertcat(0, omega_B.T / 2),
                     q_WB * vertcat(0,0,0,sum1(thrust)) / m * vertcat(q_WB[0], -q_WB[1], -q_WB[2], -q_WB[3]) + vertcat(0,0,g),
                     inv(J) * cross((P * thrust - omega_B) , (J*omega_B) ),
                     )
    

    

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = thrust
    model.p = params
    model.name = model_name

    return model