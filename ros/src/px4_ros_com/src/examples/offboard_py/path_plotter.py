import rclpy
import traceback
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry
import collections
import functions
import numpy as np

class PathPlotter(Node):
    def __init__(self) -> None:
        super().__init__('path_plotter')
        
        
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.plot_coordinates, qos_profile)
        self.reverence_subscriber = self.create_subscription(
            Vector3, 'reference_traj', self.update_ref, qos_profile)
        
        buffer_length = 100
        
        self.x_coords = collections.deque(maxlen=buffer_length)
        self.y_coords = collections.deque(maxlen=buffer_length)
        self.z_coords = collections.deque(maxlen=buffer_length)
        
        self.x_coords_ref = collections.deque(maxlen=buffer_length)
        self.y_coords_ref = collections.deque(maxlen=buffer_length)
        self.z_coords_ref = collections.deque(maxlen=buffer_length)

        self.fig, self.ax = plt.subplots()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.line = self.ax.plot([], [], [], 'r-')[0]
        self.line_ref = self.ax.plot([], [], [], 'b-')[0]
        plt.legend()
        plt.ion()
        plt.show()
    def update_ref(self, msg):
        
        
        self.x_coords_ref.append(msg.x)
        self.y_coords_ref.append(msg.y)
        self.z_coords_ref.append(msg.z)
        
    def plot_coordinates(self, msg):
        pos = functions.NED_to_ENU(msg.position)
        
        
        self.x_coords.append(pos[0])
        self.y_coords.append(pos[1])
        self.z_coords.append(pos[2])
        
        
        
        self.line.set_data(self.x_coords, self.y_coords)
        self.line.set_3d_properties(self.z_coords)
        self.line_ref.set_data(self.x_coords_ref, self.y_coords_ref)
        self.line_ref.set_3d_properties(self.z_coords_ref)

        self.ax.set_xlim(-6, 2)
        self.ax.set_ylim(-4, 4)
        self.ax.set_zlim(0, 5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None) -> None:
    rclpy.init(args=args)
    path_plotter = PathPlotter()
    print('start')
    rclpy.spin(path_plotter)
    path_plotter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
        print(e)