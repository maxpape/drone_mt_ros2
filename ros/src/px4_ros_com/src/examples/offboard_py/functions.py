import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
from casadi import SX, vertcat, Function, sqrt, norm_2, dot, cross, atan2, if_else
import spatial_casadi as sc


def euler_to_quaternion_numpy(rpy):
    """
    Convert Euler angles to quaternion.

    Parameters:
    rpy : np.ndarray roll, pitch, yaw

    Returns:
    np.ndarray
        Quaternion [w, x, y, z] representing the rotation.
    """
    roll, pitch, yaw = rpy
    # Create a rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    
    # Convert the rotation object to quaternion (scalar-last format)
    q = r.as_quat()
    
    
    return np.array([q[3], q[0], q[1], q[2]] ) / np.linalg.norm(q)

def quaternion_to_euler_numpy(q):
    """Convert quaternion to euler angles

    Args:
        q (np.ndarray): Input quaternion 

    Returns:
        no.ndarray: Array containg orientation in euler angles [roll, pitch, yaw]
    """
    quat = np.zeros(4)
    quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]

    rotation = R.from_quat(quat)

    return rotation.as_euler("xyz", degrees=True)

def quaternion_to_euler_casadi(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x-axis, pitch is rotation around y-axis,
        and yaw is rotation around z-axis.
        """
        quat = SX.sym("quat", 4)
        quat[0], quat[1], quat[2], quat[3] = q[1], q[2], q[3], q[0]
        rotation = sc.Rotation.from_quat(quat)
        
        
        return rotation.as_euler('xyz')



def quaternion_inverse_numpy(q):
    """Invert a quaternion given as a numpy expression

    Args:
        q (np.ndarray): input quaternion

    Returns:
        np.ndarray: inverted quaternion
    """

    return np.array([1, -1, -1, -1]) * q / (np.linalg.norm(q) ** 2)

def quaternion_product_numpy(q, p):
    """Multiply two quaternions given as numpy arrays

    Args:
        q (np.ndarray): input quaternion q
        p (np.ndarray): input quaternion p

    Returns:
        np.ndarray: output quaternion
    """
    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return np.concatenate((s, v), axis=None)

def quat_rotation_numpy(v, q):
    """Rotates a vector v by the quaternion q

    Args:
        v (np.ndarray): input vector
        q (np.ndarray): input quaternion

    Returns:
        np.ndarray: rotated vector
    """

    p = np.concatenate((0.0, v), axis=None)
    p_rotated = quaternion_product_numpy(
        quaternion_product_numpy(q, p), quaternion_inverse_numpy(q)
    )
    return p_rotated[1:4]


def multiply_quaternions_casadi(q, p):
    """Multiply two quaternions given as casadi expressions

    Args:
        q (casadi SX): quaternion 1
        p (casadi SX): quaternion 2

    Returns:
        casadi SX: resulting quaternion
    """
    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - dot(v1, v2)
    v = s1 * v2 + s2 * v1 + cross(v1, v2)
    return vertcat(s, v)

def quaternion_inverse_casadi(q):
    """Invert a quaternion given as a casadi expression

    Args:
        q (casadi SX): input quaternion

    Returns:
        casadi SX: inverted quaternion
    """

    return SX([1, -1, -1, -1]) * q / (norm_2(q)**2)

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

def quaternion_error_casadi(q_ref, q):
    """Calculate the quaternion error between a reference quaternion q_ref and an origin quaternion q

    Args:
        q_ref (casadi SX): reference quaternion
        q (casadi SX): origin quaternion

    Returns:
        casadi SX: elements x, y, and z from error quaternion (w neglected, since norm(unit quaternion)=1; not suitable for error calculation)
    """
    q_error = multiply_quaternions_casadi(q_ref, quaternion_inverse_casadi(q))

    if_else(q_error[0] >= 0, SX([1, -1, -1, -1])*q_error, SX([1, 1, 1, 1])*q_error, True)
    

    return q_error[1:4]


def quaternion_error_numpy(q_ref, q):
    """Calculate the quaternion error between a reference quaternion q_ref and an origin quaternion q

    Args:
        q_ref (casadi SX): reference quaternion
        q (casadi SX): origin quaternion

    Returns:
        casadi SX: elements x, y, and z from error quaternion (w neglected, since norm(unit quaternion)=1; not suitable for error calculation)
    """
    q_error = quaternion_product_numpy(q_ref, quaternion_inverse_numpy(q))

    if q_error[0] >= 0:
        q_error = np.array([1, -1, -1, -1])*q_error
    else:
        q_error = np.array([1, 1, 1, 1])*q_error
    

    return q_error


def NED_to_ENU(input_array):
    """
    Convert NED frame convention from drone into ENU frame convention used in MPC

    Parameters:
    input_array (np.ndarray): The input to be rotated. Should be a NumPy array of shape (3,) for a 3D vector or (4,) for a quaternion.

    Returns:
    np.ndarray: The rotated 3D vector or quaternion.
    """
    
    if input_array.shape == (3,):
        # Handle as a 3D vector
        # A 180-degree rotation around the x-axis flips the signs of the y and z components
        rot_x = R.from_euler('x', 180, degrees=True)
        rotated_array = rot_x.apply(input_array)
        
        #rotated_array = rot_z.apply(rotated_array)
    elif input_array.shape == (4,):
        
        # Handle as a quaternion
        # A 180-degree rotation around the x-axis flips the signs of the y and z components
        
        input_array = input_array/np.linalg.norm(input_array)
        rotated_array = np.zeros(4)
        
        rotated_array[0] = input_array[0]
        rotated_array[1] = input_array[1]
        rotated_array[2] = -input_array[2]
        rotated_array[3] = -input_array[3]        
        
    else:
        raise ValueError("Input array must be either a 3D vector or a quaternion (shape (3,) or (4,)).")
    
    return rotated_array