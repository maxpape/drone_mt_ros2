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
from casadi import SX, vertcat, horzcat, sum1, inv, cross, mtimes 

def quat_rotation(quaternion, vector):
    """
    Rotates a 3D vector by a quaternion using CasADi SX vectors.

    Parameters:
    quaternion (ca.SX): The quaternion as an SX vector [w, x, y, z].
    vector (ca.SX): The 3D vector to be rotated as an SX vector [x, y, z].

    Returns:
    ca.SX: The rotated 3D vector as an SX vector [x, y, z].
    """
    # Extract quaternion components
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    # Quaternion to rotation matrix
    R = SX.zeros(3, 3)
    R[0, 0] = 1 - 2*y**2 - 2*z**2
    R[0, 1] = 2*x*y - 2*z*w
    R[0, 2] = 2*x*z + 2*y*w
    R[1, 0] = 2*x*y + 2*z*w
    R[1, 1] = 1 - 2*x**2 - 2*z**2
    R[1, 2] = 2*y*z - 2*x*w
    R[2, 0] = 2*x*z - 2*y*w
    R[2, 1] = 2*y*z + 2*x*w
    R[2, 2] = 1 - 2*x**2 - 2*y**2

    # Rotate the vector
    rotated_vector = mtimes(R, vector)

    return rotated_vector


# Function to compute the quaternion product matrix for quaternion multiplication
def quaternion_product_matrix(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return vertcat(horzcat(w, -x, -y, -z),
                    horzcat(x,  w, -z,  y),
                    horzcat(y,  z,  w, -x),
                    horzcat(z, -y,  x,  w))

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
    jxx = SX.sym('jxx')
    jyy = SX.sym('jyy')
    jzz = SX.sym('jzz')

    d_x0 = SX.sym('d_x0')
    d_x1 = SX.sym('d_x1')
    d_x2 = SX.sym('d_x2')
    d_x3 = SX.sym('d_x3')
    d_y0 = SX.sym('d_y0')
    d_y1 = SX.sym('d_y1')
    d_y2 = SX.sym('d_y2')
    d_y3 = SX.sym('d_y3')
    c_tau = SX.sym('c_tau')


    params = vertcat(m, g, jxx, jyy, jzz, d_x0, d_x1, d_x2, d_x3, d_y0, d_y1, d_y2, d_y3, c_tau)

    J = vertcat(horzcat(jxx, 0, 0), horzcat(0, jyy, 0), horzcat(0, 0, jzz))  # Inertia matrix
    P = vertcat(horzcat(-d_x0, -d_x1, d_x2, d_x3), horzcat(d_y0, -d_y1, -d_y1, d_y3), horzcat(-c_tau, c_tau, -c_tau, c_tau))

    # xdot
    p_WB_dot = SX.sym('p_WB_dot', 3)        # derivative of Position of the quadrotor (x, y, z)
    q_WB_dot = SX.sym('q_WB_dot', 4)        # derivative of Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    v_WB_dot = SX.sym('v_WB_dot', 3)        # derivative of Linear velocity of the quadrotor
    omega_B_dot = SX.sym('omega_B_dot', 3)  # derivative of Angular velocity of the quadrotor in body frame

    xdot = vertcat(p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot)



    f_expl = vertcat(v_WB,
                    0.5 * mtimes(quaternion_product_matrix(q_WB), vertcat(0, omega_B)),
                    quat_rotation(q_WB, vertcat(0,0,sum1(thrust))) / m + vertcat(0,0,g),
                    mtimes(inv(J) , (mtimes(P , thrust) - cross( omega_B , mtimes(J,omega_B)) )),
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