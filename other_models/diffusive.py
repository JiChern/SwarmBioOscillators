
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

import numpy as np  # Import NumPy for numerical operations
import math  # Redundant import of math (already imported as m)
import rospy  # Import ROS for node initialization, publishers, subscribers
from numpy.random import random  # Import random function from NumPy


# Class for Diffusive Central Pattern Generator (CPG) for generating oscillatory signals
class DiffusiveCPG(object):
    # Constructor to initialize CPG with parameters
    def __init__(self,cell_num, alpha, beta, mu, omega, gamma, hz):
        self.cell_num = cell_num  # Number of oscillator cells
        self.alpha = alpha  # Hopf oscillator parameter
        self.beta = beta  # Hopf oscillator parameter
        self.mu = mu  # Hopf oscillator parameter (amplitude squared)
        self.omega = omega  # Angular frequency
        self.gamma = gamma  # Coupling strength
        self.dt = 1/hz  # Time step based on frequency


        self.theta = np.zeros(6)  # Phase offsets (fixed to 6, but cell_num may vary; potential inconsistency)
        self.phase = np.zeros(6)  # Phases (fixed to 6)

  

    # Compute rotation matrix (clockwise? but named as is)
    def rotation_mat(self, theta):
        rotation_mat = np.array([[math.cos(theta), math.sin(theta)],
                                 [-math.sin(theta), math.cos(theta)]])
        return rotation_mat

    # Compute counter-clockwise rotation matrix
    def rotation_mat_cc(self, theta):
        rotation_mat = np.array([[math.cos(theta), -math.sin(theta)],
                                 [math.sin(theta), math.cos(theta)]])
        return rotation_mat



    # Hopf oscillator dynamics for a single cell
    def hopf(self, x,y):
        
        r_2 = x ** 2 + y ** 2  # Squared radius
        dx = self.alpha * (self.mu - r_2) * x - self.omega * y  # x-dot
        dy = self.beta * (self.mu - r_2) * y + self.omega * x  # y-dot

        return np.array([dx,dy]).reshape(2,1)    


    # Set phase offsets
    def set_theta(self, theta):
        self.theta = theta


    # Reset initial states of oscillators
    def reset_ini_states(self, r):
        """Reset the environment to an initial state.
        
        Args:
            ini_state (np.ndarray, optional): Initial state of oscillators (shape: [cell_nums, 2]). 
                If None, samples random initial states. Defaults to None.
        
        Returns:
            np.ndarray: Initial observation (shape: [cell_nums * 2 + cell_nums]).
        """

        # use this before reset_goal function

        # Random initial state if not provided

        z_x = np.random.uniform(-r,r, self.cell_num)  # Random x in [-r, r]
        radius = r*np.ones(self.cell_num)  # Fixed radius for each
        z_y = np.sqrt(radius*radius-z_x*z_x)  # Compute y for circle
        z_y = np.where(random(self.cell_num) > 0.5, -z_y, z_y)  # Randomly flip sign for y 
        self.pos = np.array([z_x,z_y])  # Set positions [x's, y's]

        return self.pos


    # Compute diffusive coupling updates
    def diffusive(self, pos):
        # Given current position of all legs and step length, Compute positions at next step
        #p os = [x1, x2, x3, x4, y1, y2, y3, y4]  # (Comment for 4 cells, but general)

        x = np.array(pos[0])  # Extract x components
        y = np.array(pos[1])  # Extract y components

        dx = np.zeros(self.cell_num)  # Initialize dx
        dy = np.zeros(self.cell_num)  # Initialize dy

        # Loop through each cell
        for i in range(self.cell_num):
            z_i = np.array([x[i],y[i]]).reshape(2,1)  # Current state vector

            
            R = self.rotation_mat_cc(self.theta[i])  # Counter-clockwise rotation matrix

            # Handle coupling: ring topology (last couples to first)
            if i<self.cell_num-1:
                coeff = 1  # Coupling coefficient
                # get the state of the next cell
                z_i_ = np.array([x[i+1],y[i+1]]).reshape(2,1)
                # perform ccw rotation of this state vector
                r_z_i_ = np.matmul(R, z_i_)


                F_zi = self.hopf(x[i+1],y[i+1])  # Hopf for next cell (potential bug: should be for current?)
                


            else:
                coeff = 1
                z_i_ = np.array([x[0],y[0]]).reshape(2,1)  # Wrap around to first
                r_z_i_ = np.matmul(R,z_i_)


            # get the first derivate outputs from internal Hopf dynamics
            F_zi = self.hopf(x[i],y[i])  # Hopf for current cell (overwritten if i<cell_num-1)

            # get the first derivate outputs from internal diffusive system
            dz_i = F_zi + coeff*self.gamma*(r_z_i_-z_i)  # Combine Hopf and diffusive coupling
            dx[i] = dz_i[0,0]  # Update dx
            dy[i] = dz_i[1,0]  # Update dy
            # print(dz_i)


     
        return x + dx*self.dt, y + dy*self.dt  # Return updated x and y



    # Update positions softly (using diffusive)
    def update_soft(self):
        
        # x,y = self.normalized_difussive_hopf_coupling_soft(self.pos, steps=steps)  # (Commented out: alternative method)
        x,y = self.diffusive(self.pos)  # Compute updates
        self.pos = np.array([x,y])  # Update positions


