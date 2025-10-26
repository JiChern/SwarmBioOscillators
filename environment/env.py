import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
# import random
from numpy.random import random

class CPGEnv(object):
    """A custom off-policy RL environment for controlling coupled oscillators with graph-structured state observations.
        
        This environment simulates a system of coupled oscillators (e.g., Hopf oscillators) where the agent's goal is to 
        adjust the relative phases (lags) between oscillators to match a desired configuration. The state includes 
        oscillator positions, and the encoding of the desired phase lags. The agent interacts with the 
        environment by applying continuous actions to the oscillators, and receives rewards based on how closely the 
        relative phases match the desired lags.

        The environment is designed for off-policy RL algorithms (e.g., TD3, SAC) and integrates with PyTorch for efficient 
        training. The state includes both oscillator positions and desired lag encoding, while actions 
        directly modulate the oscillators' dynamics.

        Attributes:
            cell_nums (int): Number of coupled oscillators in the system.
            env_length (int): Maximum number of steps per episode.
            damp (float): Damping coefficient for oscillator dynamics (default: 0).
            x (np.ndarray): Current x-positions of oscillators (shape: [cell_nums]).
            v (np.ndarray): Current velocities of oscillators (shape: [cell_nums]).
            v_dot (np.ndarray): Current accelerations of oscillators (shape: [cell_nums]).
            dt (float): Time step for simulation (default: 0.01).
            z_mat (np.ndarray): Matrix of oscillator states (shape: [cell_nums, 2]; columns: x, y).
            desired_lag_list (np.ndarray): Predefined list of desired phase lag configurations (shape: [4, 8]).
            row_index (np.ndarray): Indices for sampling desired lag configurations (shape: [4]).
            prob (np.ndarray): Probability distribution for sampling desired lag configurations (uniform).
            desired_lag (np.ndarray): Currently active desired phase lags (shape: [cell_nums]).
            relative_lags (np.ndarray): Relative phase lags between oscillators (shape: [cell_nums, cell_nums]).
            reward_thres (float): Reward threshold for early termination (default: 2e-1).
            internal_step (int): Current step count within the episode.
    """

    def __init__(self, cell_nums, env_length, hz=None):
        """Initialize the CPG environment with given parameters.
        
        Args:
            cell_nums (int): Number of coupled oscillators.
            env_length (int): Maximum episode length (steps).
            hz (int): Loop Frequency
        """
        self.damp = 0
        self.x = np.ones(6)
        self.v = np.zeros(6)
        self.v_dot = np.zeros(6)

        if hz is None:
            self.dt = 0.01
        else:
            self.dt = 1/hz
        self.z_mat = np.array([[0.1,0,0,0],
                              [0,0,0,0]]).transpose()
        pi = np.pi
        
        self.cell_nums = cell_nums

        self.env_length = env_length

        # Predefined desired lag configurations (e.g., for 8 oscillators)
        self.desired_lag_list = np.array([[0,0.5,0.25,0.75,0.5,0,0.75,0.25],              
                                            [0,0,0.75,0.75,0.5,0.5,0.25,0.25],       
                                            [0,0.5,0.5,0,0,0.5,0.5,0],
                                            [0,0,0.5,0.5,0,0,0.5,0.5]    
                                            ])

        # Sampling distribution for desired lag configurations
        self.row_index = np.array([0,1,2,3])
        self.prob = np.array([1/4,1/4,1/4,1/4])
        index = np.random.choice(self.row_index,p=self.prob)


        # Initialize desired lag and relative lags
        # self.desired_lag = self.desired_lag_list[index,:]
        # self.relative_lags = self.cal_relative_lags(self.desired_lag)


        self.reward_thres = 2e-1
        self.internal_step = 0


    def cal_relative_lags(self, lag_list):
        """Compute relative phase lags between all oscillator pairs.
        
        Args:
            lag_list (np.ndarray): Desired phase lags for each oscillator (shape: [cell_nums]).
        
        Returns:
            np.ndarray: Relative lags matrix (shape: [cell_nums, cell_nums]).
        """
        relative_lags = np.zeros((self.cell_nums,self.cell_nums))
        for i in range(self.cell_nums):
            for j in range(self.cell_nums):
                relative_lags[i,j] = lag_list[i]-lag_list[j]
        return relative_lags
    
    def reset_goal(self, index = None):
        """Reset the desired phase lag configuration.
        
        Args:
            index (int, optional): Index of the desired lag configuration to use. 
                If None, samples randomly from the predefined list. Defaults to None.
        
        Returns:
            tuple[np.ndarray, int]: 
                - rl_encoding (np.ndarray): Sin/Cos encoding of the desired lags (shape: [cell_nums, 2]).
                - index (int): Index of the selected desired lag configuration.
        """

        if index is None:
            index = np.random.choice(self.row_index,p=self.prob)

        self.desired_lag = self.desired_lag_list[index,:]
        self.relative_lags = self.cal_relative_lags(self.desired_lag)

        rl_encoding = self.encoding_angle(self.desired_lag)

        return rl_encoding.ravel(), index


    def reset(self, ini_state=None):
        """Reset the environment to an initial state.
        
        Args:
            ini_state (np.ndarray, optional): Initial state of oscillators (shape: [cell_nums, 2]). 
                If None, samples random initial states. Defaults to None.
        
        Returns:
            np.ndarray: Initial observation (shape: [cell_nums * 2 + cell_nums]).
        """

        # use this before reset_goal function

        # Random initial state if not provided
        if ini_state is None:
            z_x = np.random.uniform(-1.2,1.2, self.cell_nums)
            radius = 1.2*np.ones(self.cell_nums)
            z_y = np.sqrt(radius*radius-z_x*z_x)
            z_y = np.where(random(self.cell_nums) > 0.5, -z_y, z_y) 
            self.z_mat = np.array([z_x,z_y]).transpose()
            obs = self.z_mat.ravel()
        
        else:
            self.z_mat = np.reshape(ini_state,(self.cell_nums,2))
            obs = self.z_mat.ravel()


        rl_encoding = self.encoding_angle(self.desired_lag)

        obs = np.concatenate((obs,rl_encoding.ravel()))

        # obs = np.concatenate((obs,rl))

        self.internal_step = 0
        return obs
        

    def rotation_mat(self,theta):
        """Compute the 2D rotation matrix for a given angle (clockwise).
        
        Args:
            theta (float): Rotation angle in radians.
        
        Returns:
            np.ndarray: 2x2 rotation matrix.
        """
        R = np.array([[np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]])
        return R
    
    def rotation_mat_ccw(self,theta):
        """Compute the 2D rotation matrix for a given angle (counter-clockwise).
        
        Args:
            theta (float): Rotation angle in radians.
        
        Returns:
            np.ndarray: 2x2 rotation matrix.
        """

        R = np.array([[np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
        return R
    
    
    def hopf(self,x,y,ax,ay):
        """Compute the derivatives for Hopf oscillator dynamics.
        
        Args:
            x (np.ndarray): Current x-positions of oscillators.
            y (np.ndarray): Current y-positions of oscillators.
            ax (np.ndarray): x-component of external action.
            ay (np.ndarray): y-component of external action.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: Derivatives of x and y positions.
        """

        alpha = 10
        beta = 10
        omega = 2*np.pi

        mu = np.ones(self.cell_nums)

        r_2 = x*x + y*y

        dx = alpha * (mu - r_2) * x - omega * y + ax
        dy = beta * (mu - r_2) * y + omega * x + ay

        return dx,dy

    def van_der_pol(self,x,y, ax,ay):
        """Compute the derivatives for Van der Pol oscillator dynamics.
        
        Args:
            x (np.ndarray): Current x-positions of oscillators.
            y (np.ndarray): Current y-positions of oscillators.
            ax (np.ndarray): x-component of external action.
            ay (np.ndarray): y-component of external action.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: Derivatives of x and y positions.
        """

        b = 4
        omega = 2*np.pi

        epsilon = 1

        ones_vec = np.ones(self.cell_nums)

        dx = omega*y + epsilon*x*(ones_vec-b*x*x/3) + ax
        dy = -omega*x + ay
        return dx, dy
    
    def damped_spring(self, x,y, ax,ay):
        """Compute the derivatives for damped spring oscillator dynamics.
        
        Args:
            x (np.ndarray): Current x-positions of oscillators.
            y (np.ndarray): Current y-positions of oscillators.
            ax (np.ndarray): x-component of external action.
            ay (np.ndarray): y-component of external action.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: Derivatives of x and y positions.
        """
        gamma = 20
        alpha = 0.0

        dy = -gamma*x-alpha*y + ay
        dx = y + ax

        return dx, dy   
    
    def encoding_angle(self, desired_lag_list):
        """Encode desired phase lags using trigonometric functions.
        
        Args:
            desired_lag_list (np.ndarray): Desired phase lags for each oscillator (shape: [cell_nums]).
        
        Returns:
            np.ndarray: Encoding matrix (shape: [cell_nums, 2]).
        """
        encoding_mat = np.zeros((self.cell_nums,2))
        for i in range(self.cell_nums):
            encoding_mat[i,0] = np.sin(desired_lag_list[i]*2*np.pi)
            encoding_mat[i,1] = np.cos(desired_lag_list[i]*2*np.pi)

        return encoding_mat

    def step_env(self, action):
        """Execute one step of the environment with given actions.
        
        Args:
            action (np.ndarray): Actions for oscillators (shape: [cell_nums, 2]; columns: x, y).
        
        Returns:
            np.ndarray: Next observation (shape: [cell_nums * 2 + cell_nums]).
        """
    
        action = np.reshape(action,(-1,2))
        action_x = action[:,0]
        action_y = action[:,1]

        

        x = self.z_mat[:,0]
        y = self.z_mat[:,1]

        dx, dy = self.hopf(x,y,action_x,action_y)

        

        self.z_mat[:,0] = x + dx*self.dt
        self.z_mat[:,1] = y + dy*self.dt

        obs = self.z_mat.ravel()
 


        rl_encoding = self.encoding_angle(self.desired_lag)
        obs = np.concatenate((obs,rl_encoding.ravel()))

        done = False
        info = None


        self.internal_step += 1

        return obs

    def step(self, action):
        """Execute one step of the environment with given actions and compute rewards.
        
        Args:
            action (np.ndarray): Actions for oscillators (shape: [cell_nums, 2]; columns: x, y).
        
        Returns:
            tuple[np.ndarray, float, bool, dict]: 
                - obs (np.ndarray): Next observation (shape: [cell_nums * 2 + cell_nums]).
                - reward (float): Scalar reward for the step.
                - done (bool): Whether the episode is terminated.
                - info (dict): Additional episode information (None in this case).
        """
    
        action = np.reshape(action,(-1,2))
        action_x = action[:,0]
        action_y = action[:,1]

        x = self.z_mat[:,0]
        y = self.z_mat[:,1]

        state = self.z_mat.ravel()

        dx, dy = self.hopf(x,y,action_x,action_y)
        

        self.z_mat[:,0] = x + dx*self.dt
        self.z_mat[:,1] = y + dy*self.dt

        next_state = self.z_mat.ravel()

        # Compute reward (negative of phase mutual error)
        reward = self.cal_reward_mutual_1()
        reward = -reward

        # Encode relative lags and concatenate to observation
        rl = self.relative_lags.ravel()
        rl_encoding = self.encoding_angle(self.desired_lag)

        obs = np.concatenate((next_state,rl_encoding.ravel()))


        done = False
        info = None

        # Termination conditions: reward threshold or max steps
        if (self.internal_step>100 and reward > -self.reward_thres) or self.internal_step>self.env_length:
            done = True
    


        self.internal_step += 1

        return obs, float(reward), done, info

           
    def cal_reward_mutual(self):
        """Compute mutual phase rewards based on relative angles between oscillators.

        Returns:
            float: Scalar reward representing phase alignment.
        """
        reward = 0


        relative_lags = self.relative_lags

        relative_angles = relative_lags*2*np.pi

        for i in range(self.cell_nums):

            z_i = np.array([[self.z_mat[i,0]],[self.z_mat[i,1]]])
            z_i_norm = z_i/(np.linalg.norm(z_i)+1e-5)
            
            # phase reward
            for j in range(self.cell_nums):

                z_j = np.array([[self.z_mat[j,0]],[self.z_mat[j,1]]])
                z_j_norm = z_j/(np.linalg.norm(z_j)+1e-5)


                R=self.rotation_mat_ccw(relative_angles[i,j])
                reward += abs(np.linalg.norm(R@z_j_norm-z_i_norm))


        reward = reward/10

        return reward


    def cal_reward_mutual_1(self):
        """Compute mutual phase rewards using vectorized operations for efficiency.
        
        Returns:
            float: Scalar reward representing phase alignment.
        """

        # Normalize all vectors at once (avoids repeated computation)
        z_norm = self.z_mat / (np.linalg.norm(self.z_mat, axis=1, keepdims=True) + 1e-5)
        
        # Precompute all rotation components
        theta = self.relative_lags * 2 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Vectorized rotation and difference calculation
        rotated_x = cos_theta * z_norm[:, 0][np.newaxis, :] - sin_theta * z_norm[:, 1][np.newaxis, :]
        rotated_y = sin_theta * z_norm[:, 0][np.newaxis, :] + cos_theta * z_norm[:, 1][np.newaxis, :]
        
        # Compute differences using broadcasting
        diff_x = rotated_x - z_norm[:, 0][:, np.newaxis]
        diff_y = rotated_y - z_norm[:, 1][:, np.newaxis]
        
        # Calculate norm for all pairs simultaneously
        reward = np.sqrt(diff_x**2 + diff_y**2).sum() / 10
        
        return reward

