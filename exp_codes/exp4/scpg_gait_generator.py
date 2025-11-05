import rospy
import torch
import torch_geometric
import sys, os
import numpy as np
import math
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from pathlib import Path  # Import Path for handling file paths (SCPG models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path for importing custom modules

from agent.networks import Policy  # Import SCPG architecture (Policy is a GNN model)
from env_test import CPGEnv  # Import CPG environment for oscillatory dynamics
from utils import rearrange_state_vector_hopf  # Utility to rearrange state vectors for Hopf oscillators
from utils import generate_edge_idx  # Utility to generate edge indices for graph structure
import matplotlib.pyplot as plt
import time
pi = np.pi  # Define pi constant for convenience

class SCPG(object):
    """
    A class implementing a Graph-based Central Pattern Generator (SCPG) for gait generation.
    Uses a GNN policy to compute actions and a CPG environment to simulate oscillatory dynamics.
    Designed to run on a remote PC for controlling a robot via ROS.
    """
    def __init__(self, network, checkpoint, env) -> None:
        """
        Initialize the SCPG system.

        Args:
            network (torch.nn.Module): GNN-based policy network (Policy class).
            checkpoint (dict): Pre-trained model checkpoint containing policy state dictionary.
            env (CPGEnv): CPG environment instance for state updates.

        Attributes:
            network: Loaded GNN policy network with pre-trained weights.
            env: CPG environment for state evolution.
            o_leg_sub: ROS subscriber for operating legs and desired phase lags.
            o_leg_list: List of active leg indices (default: [0,1,2,3,4,5]).
            cell_nums: Number of cells (legs) in the system.
            desired_lag: Desired phase lags for each leg (default: [0,0.5,0,0.5,0,0.5]).
        """
        self.network = network
        self.network.load_state_dict(checkpoint['policy_state_dict'])  # Load pre-trained weights
        self.env = env

        # Subscribe to ROS topic 'operating_legs' for dynamic leg and phase lag updates
        self.o_leg_sub = rospy.Subscriber('operating_legs', Float32MultiArray, self.o_leg_cb)
        self.o_leg_list = [0, 1, 2, 3, 4, 5]  # Default: 6 legs, indexed 0 to 5
        self.cell_nums = len(self.o_leg_list)  # Number of cells (legs)
        self.desired_lag = [0, 1/2, 0, 1/2, 0, 1/2]  # Default phase lags (alternating for gait)

    def o_leg_cb(self, data):
        """
        Callback for the 'operating_legs' ROS topic to update active legs and phase lags.

        Args:
            data (Float32MultiArray): ROS message containing leg indices and desired phase lags.

        Notes:
            - The message data is split into leg indices (first dim1_size elements) and phase lags (remaining).
            - Updates self.o_leg_list, self.desired_lag, and self.cell_nums dynamically.
        """
        dim1_size = data.layout.dim[0].size  # Size of leg indices
        dim2_size = data.layout.dim[1].size  # Size of phase lags
        self.o_leg_list = data.data[:dim1_size]  # Update active leg indices
        self.desired_lag = data.data[dim1_size:]  # Update desired phase lags
        self.cell_nums = len(self.o_leg_list)  # Update number of cells

    def encoding_angle(self, desired_lag_list):
        """
        Encode desired phase lags into a matrix of (sin, cos) pairs for each cell.

        Args:
            desired_lag_list (list or numpy.ndarray): List of phase lags (in cycles, e.g., 0 to 1) for each cell.

        Returns:
            numpy.ndarray: Matrix of shape (cell_nums, 2) where each row is [sin(2π*lag), cos(2π*lag)].

        Notes:
            - Used to represent phase relationships for the Hopf oscillator states.
        """
        encoding_mat = np.zeros((self.cell_nums, 2))
        for i in range(self.cell_nums):
            encoding_mat[i, 0] = np.sin(desired_lag_list[i] * 2 * np.pi)
            encoding_mat[i, 1] = np.cos(desired_lag_list[i] * 2 * np.pi)
        return encoding_mat

    def get_gnnx(self):
        """
        Prepare input for the GNN by combining current states and desired phase encodings.

        Returns:
            torch.Tensor: Rearranged state vector compatible with the GNN input format.

        Notes:
            - Uses current environment states (env.z_mat) for active cells and desired phase encodings.
            - Calls rearrange_state_vector_hopf to format the input for the GNN.
        """
        env_z = self.env.z_mat  # Current (x, y) states of all cells
        activated_cells = np.array(self.o_leg_list, dtype=int)  # Active cell indices
        activated_states = env_z[activated_cells]  # States of active cells
        rl_encoding = self.encoding_angle(self.desired_lag)  # Phase encodings
        a_nextstate = np.concatenate((activated_states.ravel(), rl_encoding.ravel()))  # Combine states and encodings
        return rearrange_state_vector_hopf(state=a_nextstate, num_nodes=self.cell_nums)  # Format for GNN

    def get_phase_data(self, state):
        """
        Compute the next state and phase angles for all cells.

        Args:
            state (numpy.ndarray): Current state vector (concatenation of env states and phase encodings).

        Returns:
            tuple: (next_state, phase)
                - next_state (numpy.ndarray): Updated state vector after one environment step.
                - phase (numpy.ndarray): Phase angles (0 to 1) for all 6 cells.

        Notes:
            - Uses the GNN to compute actions, updates the environment, and calculates phase angles.
            - Assumes 6 cells for phase output (hardcoded).
        """
        # Generate graph adjacency matrix (edge indices) for the GNN
        edge_index = generate_edge_idx(cell_num=self.cell_nums)

        # Get GNN input and compute actions
        gnn_x = self.get_gnnx()
        with torch.no_grad():  # Disable gradient computation for inference
            action = self.network(gnn_x, edge_index)  # Compute actions using GNN
            action.clamp_(-1, 1)  # Clamp actions to [-1, 1]
            action = action.squeeze().cpu().numpy()  # Convert to numpy array

        # Update environment with computed actions
        obs = self.env.step_env_simple(action, self.o_leg_list)

        # Calculate phase angles for all 6 cells
        phase = np.zeros(6)
        for i in range(6):
            phase[i] = self.cal_phase(obs[2*i], obs[2*i+1])

        # Compute next state by combining observations and phase encodings
        rl_encoding = self.encoding_angle(self.desired_lag)
        nextstate = np.concatenate((obs, rl_encoding.ravel()))

        return nextstate, phase

    def cal_phase(self, x, y):
        """
        Calculate the phase angle (0 to 1) for a cell based on its (x, y) state.

        Args:
            x (float): x-coordinate of the oscillator state.
            y (float): y-coordinate of the oscillator state.

        Returns:
            float or None: Phase angle (0 to 1) or None if x = y = 0.

        Notes:
            - Phase is used for stance-swing trajectory generation:
              - 0 to 0.5: Swing phase
              - 0.5 to 1: Stance phase
            - Computes the angle using arctangent and normalizes to [0, 1].
        """
        if x > 0 and y >= 0:
            theta = math.atan(y / x)
        elif x < 0:
            theta = math.atan(y / x) + math.pi
        elif x > 0 and y < 0:
            theta = math.atan(y / x) + 2 * math.pi
        elif x == 0 and y > 0:
            theta = math.pi / 2
        elif x == 0 and y < 0:
            theta = -math.pi / 2
        elif x == 0 and y == 0:
            theta = None
        else:
            theta = None  # Fallback case (should not occur)

        phase = theta / (2 * math.pi) if theta is not None else None
        return phase

if __name__ == '__main__':
    """
    Main execution block for running the SCPG system on a remote PC.
    Initializes ROS, sets up the CPG environment and GNN model, and runs the gait generation loop.
    Publishes gait patterns to ROS topics and plots state trajectories.
    """
    rospy.init_node('scpg') 

    hz = 50  # Frequency of updates (50 Hz)

    # Initialize SCPG system (8 attention heads)
    heads = 8  
    fd = 64    
    cell_num = 6  
    model = Policy(heads=heads, feature_dim=fd)  # Create GNN policy model
    model.eval()  
    checkpoint = torch.load(parent_dir + '/model_params/model-8-64.pt', weights_only=True)  # Load pre-trained weights
    model.load_state_dict(checkpoint['policy_state_dict'])
    env = CPGEnv(cell_nums=cell_num, env_length=500, hz=50) 
    scpg = SCPG(network=model, checkpoint=checkpoint, env=env)  

    r = rospy.Rate(hz)  # Set ROS loop rate to 50 Hz
    ei = generate_edge_idx(cell_num=cell_num)  # Generate edge indices for GNN

    # Initialize environment state (z_mat)
    env.z_mat = np.zeros((cell_num, 2))  
    env.z_mat[0, 0] = 0.1  
    obs = env.z_mat.ravel()  
    rl_encoding = scpg.encoding_angle(scpg.desired_lag) 
    state = np.concatenate((obs, rl_encoding.ravel()))  

    # Lists to store state trajectories for plotting
    x1, x2, x3, x4, x5, x6 = [], [], [], [], [], []

    # Initialize ROS publishers for gait data
    all_gait_pub = rospy.Publisher('scpg_gait_all', Float32MultiArray, queue_size=10)  # Publish all phases
    gait_pub = rospy.Publisher('scpg_gait', Float32MultiArray, queue_size=10)  # Publish active cell phases
    gait_msg = Float32MultiArray()
    all_gait_msg = Float32MultiArray()
    all_gait_msg.data = np.zeros(6)  # Initialize message for all 6 phases

    while not rospy.is_shutdown():
        """
        Main loop: Update states, compute phases, publish gait data, and log states.
        Runs at 50 Hz until ROS shutdown.
        """
        start_time = time.time()  # Track loop execution time

        # Print current operating legs and phase lags
        print('operating legs: ', scpg.o_leg_list)
        print('desired phase lags: ', scpg.desired_lag)


        # Compute next state and phase angles
        next_state, phase = scpg.get_phase_data(state)

        # Publish phases for active cells
        activated_cells = np.array(scpg.o_leg_list, dtype=int)
        activated_phases = phase[activated_cells]
        gait_msg.data = activated_phases
        gait_pub.publish(gait_msg)

        # Publish phases for all cells
        all_gait_msg.data = phase
        all_gait_pub.publish(all_gait_msg)

        state = next_state  # Update state for next iteration

        # Log state trajectories for plotting
        x1.append(next_state[0])
        x2.append(next_state[2])
        x3.append(next_state[4])
        x4.append(next_state[6])
        x5.append(next_state[8])
        x6.append(next_state[10])

        r.sleep()  # Maintain loop rate (50 Hz)
        print(time.time() - start_time)  # Print loop execution time

    # Plot state trajectories for all cells
    plt.figure()
    plt.plot(x1, label='x0')
    plt.plot(x2, label='x1')
    plt.plot(x3, label='x2')
    plt.plot(x4, label='x3')
    plt.plot(x5, label='x4')
    plt.plot(x6, label='x5')
    plt.legend()
    plt.show()
