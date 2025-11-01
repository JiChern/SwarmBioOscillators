import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
from numpy.random import random  # Can be removed if NumPy random is not needed

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
            x (torch.Tensor): Current x-positions of oscillators (shape: [cell_nums]).
            v (torch.Tensor): Current velocities of oscillators (shape: [cell_nums]).
            v_dot (torch.Tensor): Current accelerations of oscillators (shape: [cell_nums]).
            dt (float): Time step for simulation (default: 0.01).
            z_mat (torch.Tensor): Matrix of oscillator states (shape: [cell_nums, 2]; columns: x, y).
            desired_lag_list (torch.Tensor): Predefined list of desired phase lag configurations (shape: [4, 8]).
            row_index (torch.Tensor): Indices for sampling desired lag configurations (shape: [4]).
            prob (torch.Tensor): Probability distribution for sampling desired lag configurations (uniform).
            desired_lag (torch.Tensor): Currently active desired phase lags (shape: [cell_nums]).
            relative_lags (torch.Tensor): Relative phase lags between oscillators (shape: [cell_nums, cell_nums]).
            reward_thres (float): Reward threshold for early termination (default: 2e-1).
            internal_step (int): Current step count within the episode.
    """
    def __init__(self, cell_nums, env_length, hz=None, omega=2*np.pi, device='cpu'):
        """Initialize the CPG environment with given parameters.
       
        Args:
            cell_nums (int): Number of coupled oscillators.
            env_length (int): Maximum episode length (steps).
            hz (int): Loop Frequency
            device (str): Device for tensors ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.damp = 0
        self.x = torch.ones(6, device=self.device)  # Note: Originally 6, may need to adjust to cell_nums
        self.v = torch.zeros(6, device=self.device)
        self.v_dot = torch.zeros(6, device=self.device)
        if hz is None:
            self.dt = 0.01
        else:
            self.dt = 1 / hz
        self.z_mat = torch.tensor([[0.1, 0, 0, 0],
                                   [0, 0, 0, 0]], device=self.device).transpose(0, 1)
        pi = np.pi  # Can be moved to Torch, but not necessary
        
        self.cell_nums = cell_nums
        self.env_length = env_length
        self.omega = omega
        # Predefined desired lag configurations (e.g., for 8 oscillators)
        self.desired_lag_list = torch.tensor([[0, 0.5, 0.25, 0.75, 0.5, 0, 0.75, 0.25],
                                              [0, 0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25],
                                              [0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0],
                                              [0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5]],
                                             dtype=torch.float32, device=self.device)
        # Sampling distribution for desired lag configurations
        self.row_index = torch.tensor([0, 1, 2, 3], device=self.device)
        self.prob = torch.tensor([1/4, 1/4, 1/4, 1/4], device=self.device)
        index = torch.multinomial(self.prob, 1).item()
        self.desired_lag = self.desired_lag_list[index, :self.cell_nums]  # Adjust to match cell_nums
        self.relative_lags = self.cal_relative_lags(self.desired_lag)
        self.reward_thres = 2e-1
        self.internal_step = 0

    def cal_relative_lags(self, lag_list):
        """Compute relative phase lags between all oscillator pairs.
       
        Args:
            lag_list (torch.Tensor): Desired phase lags for each oscillator (shape: [cell_nums]).
       
        Returns:
            torch.Tensor: Relative lags matrix (shape: [cell_nums, cell_nums]).
        """
        # Vectorized version, more efficient
        return lag_list.unsqueeze(1) - lag_list.unsqueeze(0)

    def reset_goal(self, index=None):
        """Reset the desired phase lag configuration.
       
        Args:
            index (int, optional): Index of the desired lag configuration to use.
                If None, samples randomly from the predefined list. Defaults to None.
       
        Returns:
            tuple[torch.Tensor, int]:
                - rl_encoding (torch.Tensor): Sin/Cos encoding of the desired lags (shape: [cell_nums * 2]).
                - index (int): Index of the selected desired lag configuration.
        """
        if index is None:
            index = torch.multinomial(self.prob, 1).item()
        self.desired_lag = self.desired_lag_list[index, :self.cell_nums]
        self.relative_lags = self.cal_relative_lags(self.desired_lag)
        rl_encoding = self.encoding_angle(self.desired_lag)
        return rl_encoding.view(-1), index
    
    def to_half(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.half())
        # Handle omega if it's a scalar or NumPy value
        if hasattr(self, 'omega') and not isinstance(self.omega, torch.Tensor):
            self.omega = torch.tensor(self.omega, dtype=torch.float16, device=self.device)
        return self

    def reset(self, ini_state=None):
        """Reset the environment to an initial state.
       
        Args:
            ini_state (torch.Tensor, optional): Initial state of oscillators (shape: [cell_nums * 2]).
                If None, samples random initial states. Defaults to None.
       
        Returns:
            torch.Tensor: Initial observation (shape: [cell_nums * 2 + cell_nums * 2]).
        """
        if ini_state is None:
            z_x = torch.empty(self.cell_nums, device=self.device).uniform_(-1.2, 1.2)
            radius = 1.2 * torch.ones(self.cell_nums, device=self.device)
            z_y = torch.sqrt(radius * radius - z_x * z_x)
            signs = torch.where(torch.rand(self.cell_nums, device=self.device) > 0.5, torch.tensor(-1.0, device=self.device), torch.tensor(1.0, device=self.device))
            z_y = z_y * signs
            self.z_mat = torch.stack([z_x, z_y], dim=1)
            obs = self.z_mat.view(-1)
        else:
            self.z_mat = ini_state.view(self.cell_nums, 2).to(self.device)
            obs = self.z_mat.view(-1)
        rl_encoding = self.encoding_angle(self.desired_lag)
        obs = torch.cat((obs, rl_encoding.view(-1)))
        self.internal_step = 0
        return obs

    def reset_ini_states(self, r, ini_state=None):
        """Reset the environment to an initial state with radius r.
       
        Args:
            r (float): Radius for initial states.
            ini_state (torch.Tensor, optional): Initial state.
       
        Returns:
            torch.Tensor: Initial observation.
        """
        if ini_state is None:
            z_x = torch.empty(self.cell_nums, device=self.device).uniform_(-r, r)
            radius = r * torch.ones(self.cell_nums, device=self.device)
            z_y = torch.sqrt(radius * radius - z_x * z_x)
            signs = torch.where(torch.rand(self.cell_nums, device=self.device) > 0.5, torch.tensor(-1.0, device=self.device), torch.tensor(1.0, device=self.device))
            z_y = z_y * signs
            self.z_mat = torch.stack([z_x, z_y], dim=1)
            obs = self.z_mat.view(-1)
        else:
            self.z_mat = ini_state.view(self.cell_nums, 2).to(self.device)
            obs = self.z_mat.view(-1)
        rl_encoding = self.encoding_angle(self.desired_lag)
        obs = torch.cat((obs, rl_encoding.view(-1)))
        self.internal_step = 0
        return obs

    def rotation_mat(self, theta):
        """Compute the 2D rotation matrix for a given angle (clockwise).
       
        Args:
            theta (float or torch.Tensor): Rotation angle in radians.
       
        Returns:
            torch.Tensor: 2x2 rotation matrix.
        """
        return torch.tensor([[torch.cos(theta), torch.sin(theta)],
                             [-torch.sin(theta), torch.cos(theta)]], device=self.device)

    def rotation_mat_ccw(self, theta):
        """Compute the 2D rotation matrix for a given angle (counter-clockwise).
       
        Args:
            theta (float or torch.Tensor): Rotation angle in radians.
       
        Returns:
            torch.Tensor: 2x2 rotation matrix.
        """
        return torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                             [torch.sin(theta), torch.cos(theta)]], device=self.device)

    def hopf(self, x, y, ax, ay):
        """Compute the derivatives for Hopf oscillator dynamics.
       
        Args:
            x, y, ax, ay (torch.Tensor): Positions and actions.
       
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Derivatives.
        """
        alpha = 10
        beta = 10
        mu = torch.ones(self.cell_nums, device=self.device)
        r_2 = x * x + y * y
        dx = alpha * (mu - r_2) * x - self.omega * y + ax
        dy = beta * (mu - r_2) * y + self.omega * x + ay
        return dx, dy

    def van_der_pol(self, x, y, ax, ay):
        """Compute the derivatives for Van der Pol oscillator dynamics."""
        b = 4
        omega = 2 * np.pi
        epsilon = 1
        ones_vec = torch.ones(self.cell_nums, device=self.device)
        dx = omega * y + epsilon * x * (ones_vec - b * x * x / 3) + ax
        dy = -omega * x + ay
        return dx, dy

    def damped_oscillator(self, x, y, ax, ay):
        """Compute the derivatives for damped oscillator dynamics."""
        zeta = 1.5
        omega = 2 * np.pi
        dx = y + ax
        dy = -omega**2 * x - 2 * zeta * omega * y + ay
        return dx, dy

    def damped_spring(self, x, y, ax, ay):
        """Compute the derivatives for damped spring oscillator dynamics."""
        gamma = 20
        alpha = 0.0
        dy = -gamma * x - alpha * y + ay
        dx = y + ax
        return dx, dy

    def encoding_angle(self, desired_lag_list):
        """Encode desired phase lags using trigonometric functions.
       
        Args:
            desired_lag_list (torch.Tensor): Desired phase lags.
       
        Returns:
            torch.Tensor: Encoding matrix (shape: [cell_nums, 2]).
        """
        encoding_mat = torch.zeros((self.cell_nums, 2), device=self.device)
        encoding_mat[:, 0] = torch.sin(desired_lag_list * 2 * np.pi)
        encoding_mat[:, 1] = torch.cos(desired_lag_list * 2 * np.pi)
        return encoding_mat


    def step_env(self, action):
        """Execute one step of the environment with given actions.
       
        Args:
            action (torch.Tensor): Actions for oscillators (shape: [cell_nums * 2]).
       
        Returns:
            torch.Tensor: Next observation.
        """
        action = action.view(-1, 2)
        action_x = action[:, 0]
        action_y = action[:, 1]
        x = self.z_mat[:, 0]
        y = self.z_mat[:, 1]
        dx, dy = self.hopf(x, y, action_x, action_y)
        self.z_mat[:, 0] = x + dx * self.dt
        self.z_mat[:, 1] = y + dy * self.dt
        obs = self.z_mat.view(-1)
        rl_encoding = self.encoding_angle(self.desired_lag)
        obs = torch.cat((obs, rl_encoding.view(-1)))
        self.internal_step += 1
        return obs

    def step_env_vdp(self, action):
        """Van der Pol version of step_env."""
        action = action.view(-1, 2)
        action_x = action[:, 0]
        action_y = action[:, 1]
        x = self.z_mat[:, 0]
        y = self.z_mat[:, 1]
        dx, dy = self.van_der_pol(x, y, action_x, action_y)
        self.z_mat[:, 0] = x + dx * self.dt
        self.z_mat[:, 1] = y + dy * self.dt
        obs = self.z_mat.view(-1)
        rl_encoding = self.encoding_angle(self.desired_lag)
        obs = torch.cat((obs, rl_encoding.view(-1)))
        self.internal_step += 1
        return obs

    def step_env_damped(self, action):
        """Damped version of step_env."""
        action = action.view(-1, 2)
        action_x = action[:, 0]
        action_y = action[:, 1]
        x = self.z_mat[:, 0]
        y = self.z_mat[:, 1]
        dx, dy = self.damped_oscillator(x, y, action_x, action_y)
        self.z_mat[:, 0] = x + dx * self.dt
        self.z_mat[:, 1] = y + dy * self.dt
        obs = self.z_mat.view(-1)
        rl_encoding = self.encoding_angle(self.desired_lag)
        obs = torch.cat((obs, rl_encoding.view(-1)))
        self.internal_step += 1
        return obs

    def step(self, action):
        """Execute one step and compute rewards.
       
        Args:
            action (torch.Tensor): Actions.
       
        Returns:
            tuple[torch.Tensor, float, bool, dict]: obs, reward, done, info.
        """
        action = action.view(-1, 2)
        action_x = action[:, 0]
        action_y = action[:, 1]
        x = self.z_mat[:, 0]
        y = self.z_mat[:, 1]
        state = self.z_mat.view(-1)
        dx, dy = self.hopf(x, y, action_x, action_y)
        self.z_mat[:, 0] = x + dx * self.dt
        self.z_mat[:, 1] = y + dy * self.dt
        next_state = self.z_mat.view(-1)
        reward = self.cal_reward_mutual_1()
        reward = -reward.item()  # Convert to float
        rl_encoding = self.encoding_angle(self.desired_lag)
        obs = torch.cat((next_state, rl_encoding.view(-1)))
        done = False
        info = None
        if (self.internal_step > 100 and reward > -self.reward_thres) or self.internal_step > self.env_length:
            done = True
        self.internal_step += 1
        return obs, reward, done, info

    def cal_reward_mutual(self):
        """Compute mutual phase rewards (loop version)."""
        reward = 0.0
        relative_lags = self.relative_lags
        relative_angles = relative_lags * 2 * np.pi
        for i in range(self.cell_nums):
            z_i = self.z_mat[i, :].view(2, 1)
            z_i_norm = z_i / (torch.norm(z_i) + 1e-5)
            for j in range(self.cell_nums):
                z_j = self.z_mat[j, :].view(2, 1)
                z_j_norm = z_j / (torch.norm(z_j) + 1e-5)
                R = self.rotation_mat_ccw(relative_angles[i, j])
                reward += torch.abs(torch.norm(R @ z_j_norm - z_i_norm))
        reward = reward / 10
        return reward

    def cal_reward_mutual_1(self):
        """Compute mutual phase rewards using vectorized operations."""
        z_norm = self.z_mat / (torch.norm(self.z_mat, dim=1, keepdim=True) + 1e-5)
        theta = self.relative_lags * 2 * np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotated_x = cos_theta * z_norm[:, 0].unsqueeze(1) - sin_theta * z_norm[:, 1].unsqueeze(1)
        rotated_y = sin_theta * z_norm[:, 0].unsqueeze(1) + cos_theta * z_norm[:, 1].unsqueeze(1)
        diff_x = rotated_x - z_norm[:, 0].unsqueeze(0)
        diff_y = rotated_y - z_norm[:, 1].unsqueeze(0)
        reward = torch.sqrt(diff_x**2 + diff_y**2).sum() / 10
        return reward