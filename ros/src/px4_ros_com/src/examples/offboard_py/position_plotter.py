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
import time



class PathPlotter(Node):
    def __init__(self) -> None:
        super().__init__('path_plotter')
        
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
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
        
        self.buffer_length = 1000
        buffer_length = self.buffer_length
        
        self.x_coords = collections.deque(maxlen=buffer_length)
        self.y_coords = collections.deque(maxlen=buffer_length)
        self.z_coords = collections.deque(maxlen=buffer_length)
        self.last_real_x = 0
        self.last_real_y = 0
        self.last_real_z = 0
        self.last_real_t = time.time()
        self.t_real = collections.deque(maxlen=buffer_length)
        
        self.x_coords_ref = collections.deque(maxlen=buffer_length)
        self.y_coords_ref = collections.deque(maxlen=buffer_length)
        self.z_coords_ref = collections.deque(maxlen=buffer_length)
        self.t_ref = collections.deque(maxlen=buffer_length)
        
        self.is_first = True
        self.first_ref = np.zeros(3)
        self.counter = 0
        
        
        self.fig, self.ax = plt.subplots()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.line = self.ax.plot([], [], [], 'r-')[0]
        self.line_ref = self.ax.plot([], [], [], 'b-')[0]
        plt.legend()
        plt.ion()
        plt.show()
    def update_ref(self, msg):
        
        if self.is_first:
            self.first_ref = np.array([msg.x, msg.y, msg.z])
        
        t = time.time()
        self.x_coords_ref.append(msg.x)
        self.y_coords_ref.append(msg.y)
        self.z_coords_ref.append(msg.z)
        self.t_ref.append(t)
        
        
        self.x_coords.append(self.last_real_x)
        self.y_coords.append(self.last_real_y)
        self.z_coords.append(self.last_real_z)
        self.t_real.append(self.last_real_t)
        
        
    def plot_coordinates(self, msg):
        pos = functions.NED_to_ENU(msg.position)
        
        self.last_real_x = pos[0]
        self.last_real_y = pos[1]
        self.last_real_z = pos[2]
        self.last_real_t = time.time()
        
 
        self.line.set_data(self.x_coords, self.y_coords)
        self.line.set_3d_properties(self.z_coords)
        self.line_ref.set_data(self.x_coords_ref, self.y_coords_ref)
        self.line_ref.set_3d_properties(self.z_coords_ref)

        self.ax.set_xlim(-10, 6)
        self.ax.set_ylim(-8, 8)
        self.ax.set_zlim(0, 5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        
        
        
        self.counter +=1
        
        
        #if diff.shape[0] == self.buffer_length:
            
        if self.counter == 200:
            self.counter = 0
            x = np.array(list(self.x_coords))
            y = np.array(list(self.y_coords))
            z = np.array(list(self.z_coords))     
            real = np.vstack((x,y,z))
            
            refx = np.zeros_like(x)
            refy = np.zeros_like(y)
            refz = np.ones_like(z)*5
            ref = np.vstack((refx,refy,refz))
            
            
            
            diff_vector = ref-real
            
            diff = np.linalg.norm(diff_vector, axis=0)
            
            mean_eu = np.round(np.mean(diff), decimals=4)
            mean_x  = np.round(np.mean(np.abs(diff_vector[0])), decimals=4)
            mean_y  = np.round(np.mean(np.abs(diff_vector[1])), decimals=4)
            mean_z  = np.round(np.mean(np.abs(diff_vector[2])), decimals=4)
            
            #print('Average deviation of {:.4f} m over last {} samples. diff x: {}, diff y: {}, diff z: {}'.format(np.mean(diff), diff.shape[0], np.mean(np.abs(diff_vector[0])), np.mean(np.abs(diff_vector[1])), np.mean(np.abs(diff_vector[2]))))
            print('{:.4f},{:.4f},{:.4f},{:.4f}'.format(mean_eu, mean_x, mean_y, mean_z) )  
            
            
            self.x_coords_ref.clear()
            self.y_coords_ref.clear()
            self.z_coords_ref.clear()
            self.t_ref.clear()
            
            
            self.x_coords.clear()
            self.y_coords.clear()
            self.z_coords.clear()
            self.t_real.clear()
        
        
        
        
        
        

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