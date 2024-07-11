#!/usr/bin/env python3

import rclpy
import traceback
from rcl_interfaces.msg import SetParametersResult
import numpy as np
import scipy.linalg
import scipy.interpolate
import time
import collections
import pandas as pd
import GPy
import functions
from casadi import SX, vertcat, Function, sqrt, norm_2, dot, cross, atan2, if_else
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splev, splprep
from scipy.integrate import quad
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleStatus, ActuatorMotors, VehicleOdometry, ActuatorOutputs, SensorCombined, VehicleLocalPosition
from geometry_msgs.msg import Vector3
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange, IntegerRange


class GP_estimator():


    def __init__(self) -> None:

        
        # GP model and parameters
        # gp parameters

        self.noise_variance_lin = 0.25
        

        self.noise_variance_ang = 0.55
        

        

        
        self.length_hypers = [[1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7]]
        self.scale_hypers = [1, 1, 1, 1, 1, 1]
        #self.length_hypers_ang = [[1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7]]
        #self.scale_hypers_ang = [1, 1, 1]
        
        
        kerns = [GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[0], lengthscale=self.length_hypers[0], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[1], lengthscale=self.length_hypers[1], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[2], lengthscale=self.length_hypers[2], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[3], lengthscale=self.length_hypers[3], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[4], lengthscale=self.length_hypers[4], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[5], lengthscale=self.length_hypers[5], active_dims=[0,1,2,3,4,5], ARD=True)]
        lower_lenght = [0.1, 0.1, 4,4,4,4, 0.1, 0.1, 0.1,0.1,0.1,0.1]
        upper_lenght = [10, 10, 20,20,20,20, 50,50,50,50,50,50]
        #lower_lenght_ang = [0.1, 0.1, 0.1,0.1,0.1,0.1]
        #upper_lenght_ang = [50,50,50,50,50,50]
        
        for i in range(6):
            for j in range(6):
                if i >= 2:
                    kerns[i].lengthscale[[j]].constrain_bounded(lower_lenght[j], upper_lenght[j])
                else:
                    kerns[i].lengthscale[[j]].constrain_bounded(lower_lenght[j+3], upper_lenght[j+3])
            #kerns[i].variance.constrain_bounded(1e-3, 5)
            kerns[i].variance.fix()
        
        
        
        
        self.models = [GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[0]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[1]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[2]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[3]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[4]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[5])]
        
        for model in self.models:
            model.Gaussian_noise.variance = self.noise_variance_lin
            model.Gaussian_noise.variance.fix()
        
        self.online_regression = False
        self.show_lin = True
        
    def predict_accel(self, x, y, new_x, axis, dim=6):
        
        
            
        self.models[axis].set_XY(x, y)
        
        for i in range(dim):
            self.models[axis].rbf.lengthscale[i] = self.length_hypers[axis][i] 
        self.models[axis].rbf.variance[0] = self.scale_hypers[axis] 
        self.models[axis].Gaussian_noise.variance = self.noise_variance_lin
        
        
        if self.online_regression :
            self.models[axis].optimize(max_iters=1)
            
            
            
        for i in range(dim):
            self.length_hypers[axis][i] = self.models[axis].rbf.lengthscale[i]
        
        self.scale_hypers[axis] = self.models[axis].rbf.variance[0]
        
        #if axis == 0 and self.show_lin:
        #    print(self.length_hypers_lin[axis])
        #    print(self.scale_hypers_lin[axis])
        #    print('-------------')
        mean, var = self.models[axis].predict(new_x)
        
        
        
            
        return mean, var
    
    
    

        

        

       