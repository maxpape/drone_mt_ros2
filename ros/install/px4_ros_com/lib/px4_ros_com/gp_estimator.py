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
        

        self.noise_variance_ang = 0.25
        
        noise_offset = 0.1
        

        
        self.length_hypers = [[1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7], [1,5,7,7,7,7]]
        self.scale_hypers = [1, 1, 1, 1, 1, 1]
        
        
        
        kerns = [GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[0], lengthscale=self.length_hypers[0], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[1], lengthscale=self.length_hypers[1], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[2], lengthscale=[2,1,1,1,1,1], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[3], lengthscale=self.length_hypers[3], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=6, variance=self.scale_hypers[4], lengthscale=self.length_hypers[4], active_dims=[0,1,2,3,4,5], ARD=True),
                 GPy.kern.RBF(input_dim=4, variance=self.scale_hypers[5], lengthscale=[1,1,1,1], active_dims=[0,1,2,3], ARD=True)]
        #lower_lenght = [2,1,1,1,1,1,2,1,1,1,1,1]
        #upper_lenght = [10, 10, 20,20,20,20, 20,20,50,50,50,50]
        
        #lower_lenght = [0.5,0.1, 0.2,0.2,0.2,0.2, 2,0.1,0.1,0.1,0.1,0.1]
        #upper_lenght = [  10,  1,   5,  5,  5,  5, 10,  1,  1,  1,  1,  1]
        
        lower_lenght = np.ones(12)*1.5
        upper_lenght = np.ones(12)*10
        
        
        for i in range(6):
            for j in range(6):
                if i <= 2:
                    kerns[i].lengthscale[[j]].constrain_bounded(lower_lenght[j], upper_lenght[j])
                    kerns[i].variance.fix()
                elif 5 > i >= 3:
                    kerns[i].lengthscale[[j]].constrain_bounded(lower_lenght[j+6], upper_lenght[j+6])
                    kerns[i].variance.fix()
            #kerns[i].variance.constrain_bounded(0.5, 5)
            
        
        
        
        
        
        #kerns[2].lengthscale[[0]].constrain_bounded(2,4)
        #kerns[2].lengthscale[[1]].constrain_bounded(1,4)
        #kerns[2].lengthscale[[2]].constrain_bounded(1,4)
        #kerns[2].lengthscale[[3]].constrain_bounded(1,4)
        #kerns[2].lengthscale[[4]].constrain_bounded(1,4)
        #kerns[2].lengthscale[[5]].constrain_bounded(1,4)
        kerns[2].variance.fix()
        
        
        
        
        
        kerns[5].lengthscale[[0]].constrain_bounded(0.5,10)
        kerns[5].lengthscale[[1]].constrain_bounded(0.5,10)
        kerns[5].lengthscale[[2]].constrain_bounded(0.5,10)
        kerns[5].lengthscale[[3]].constrain_bounded(0.5,10)
        kerns[5].variance.fix()
        
        
        
        ##for i in range(6):
        ##    kerns[2].lengthscale[[i]].constrain_bounded(lower_lenght_z[i], upper_lenght_z[i])
        
        
        self.models = [GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[0]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[1]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[2]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[3]),
                       GPy.models.GPRegression(np.ones((1,6)), np.ones((1,1)), kerns[4]),
                       GPy.models.GPRegression(np.ones((1,4)), np.ones((1,1)), kerns[5])]
        
        for i in range(len(self.models)):
            self.models[i].Gaussian_noise.variance = self.noise_variance_lin
            #if i <=2:
                #self.models[i].Gaussian_noise.variance.constrain_bounded(self.noise_variance_lin-noise_offset,self.noise_variance_lin+noise_offset)
            #else:
                #self.models[i].Gaussian_noise.variance.constrain_bounded(self.noise_variance_ang - noise_offset, self.noise_variance_ang + noise_offset)
            self.models[i].Gaussian_noise.variance.fix()
        
        self.online_regression = False
        self.show_lin = True
        
    def predict_accel(self, x, y, new_x, axis, optimize, dim=6):
        
        if axis == 5:
            x = x[:,2:]
            new_x = new_x[:,2:]
        if axis == 2 or axis >= 3:
            min_val = np.min(y)
            max_val = np.max(y)
            # Step 2: Calculate scale and offset
            scale = 2 / (max_val - min_val)
            offset = - (min_val * scale + 1)
            
            y = y * scale + offset
        
        
        
        #if axis == 2:
        #    scale = np.max(y) - np.min(y)
        #    y = y / scale
        #    x = np.hstack((x[:,:1], np.sum(x[:,2:], axis=1).reshape(-1,1)))
        #    new_x = np.hstack((new_x[:,:1], np.sum(new_x[:,2:], axis=1).reshape(-1,1))) 
        
        
        self.models[axis].set_XY(x, y)
        
        #for i in range(dim):
        #    self.models[axis].rbf.lengthscale[i] = self.length_hypers[axis][i] 
        #self.models[axis].rbf.variance[0] = self.scale_hypers[axis] 
        if axis <= 2 :
            self.models[axis].Gaussian_noise.variance = self.noise_variance_lin
        else:
            if axis == 5:
                self.models[axis].Gaussian_noise.variance = self.noise_variance_ang
            self.models[axis].Gaussian_noise.variance = self.noise_variance_ang
        
        lengthscale =[]
        if optimize :
            self.models[axis].optimize(max_iters=1, optimizer='scg')
            
        
            
        for i in range(dim):
            lengthscale.append(self.models[axis].rbf.lengthscale[i])
        
        #self.scale_hypers[axis] = self.models[axis].rbf.variance[0]
        
        #if axis == 0 :
         #   print(self.length_hypers[3])
          #  print(self.length_hypers[5])
           # print('------------------')
        #print(self.models[2].rbf.lengthscale[0])
            #print(self.scale_hypers[axis])
            #print('-------------')
        mean, var = self.models[axis].predict(new_x)
        
        if axis == 2 or axis >= 3:
            mean = (mean - offset) / scale
            
        return mean, var, lengthscale
    
    
    

        

        

       