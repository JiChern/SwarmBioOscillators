import torch
import sys, os
from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import PolicyV2 # import graph-CPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx, phase_distance, state_to_goal1

import numpy as np
from matplotlib.pyplot import savefig
from numpy.random import random
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reset(env):
    """
    Reset the CPG environment to a random initial state.

    Args:
        env (CPGEnv): The CPG environment instance.

    Returns:
        numpy.ndarray: Initial observation vector, combining states and desired phase encodings.
    
    Notes:
        - Initializes states on a unit circle for oscillatory stability (4 cells hardcoded).
        - Concatenates flattened states with phase encodings for RL-compatible observation.
    """
    z_x = np.random.uniform(-1, 1, 4)  # Sample random x-coordinates for 4 cells
    radius = 1 * np.ones(4)  # Unit radius for circle
    # Compute y-coordinates to lie on the unit circle, with random sign flip
    z_y = np.sqrt(radius * radius - z_x * z_x)
    z_y = np.where(random(4) > 0.5, -z_y, z_y)
    env.z_mat = np.array([z_x, z_y]).transpose()  # Set environment state matrix
    obs = env.z_mat.ravel()  # Flatten state matrix
    
    # Encode desired phase lags and concatenate to form full observation
    rl_encoding = env.encoding_angle(env.desired_lag)
    obs = np.concatenate((obs, rl_encoding.ravel()))
    env.internal_step = 0  # Reset internal step counter

    return obs


def get_train_error(cell_num, edge_index, model, env):
    """
    Compute training errors (cumulative rewards) for multiple gait targets and random seeds.

    Args:
        cell_num (int): Number of cells (nodes) in the graph.
        edge_index (torch.Tensor): Edge indices for the graph.
        model (Graph-CPG): Trained GNN policy model.
        env (CPGEnv): CPG environment instance.

    Returns:
        numpy.ndarray: Array of cumulative rewards for all targets and seeds.
    

    """
    # Define training target phase lag patterns (4 different gaits for 4 cells)
    env.desired_lag_list = np.array([[0, 0.5, 0, 0.5],       # Trot        
                                     [0, 0.5, 0.5, 0],       # Bound
                                     [0, 0, 0.5, 0.5],
                                     [0, 0.25, 0.5, 0.75]])

    reward_list = np.array([])  # Accumulate all rewards

    # Loop over each target gait
    for target in range(4):
        env.desired_lag = env.desired_lag_list[target]  # Set current target phase lags
        reward_sum_list = np.zeros(10)  # Rewards for 10 random seeds per target

        # Run 10 episodes per target
        for s in range(10):
            state = reset(env)  # Reset environment to random initial state
            
            # Simulate 250 steps
            for p in range(250):
                # Compute relative phase lags (for environment step)
                env.relative_lags = env.cal_relative_lags(env.desired_lag)
                # Prepare GNN input from state
                gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)
    
                # Compute action using the model (no gradients for evaluation)
                with torch.no_grad():
                    action = model(gnn_x, edge_index)
                    action.clamp_(-1, 1)  # Clamp actions to [-1, 1]
                    action = action.squeeze().cpu().numpy()  # Convert to numpy array
                
                # Step the environment and accumulate reward
                nextstate, reward, _, _ = env.step(action)
                # reward_sum += reward  # Commented; unused variable
                reward_sum_list[s] += reward  # Accumulate per seed
                state = nextstate  # Update state

        # Append rewards for this target to the list
        reward_list = np.concatenate((reward_list, reward_sum_list))
        
    return reward_list


if __name__ == '__main__':
    """
    Main execution block: Set up model and environment, evaluate checkpoints, and save training rewards to CSV.
    
    Notes:
        - Evaluates model checkpoints from 5k to 300k steps (in 5k increments).
        - Computes rewards for predefined gait targets to generate training performance metrics.
        - Saves cumulative rewards per checkpoint to 'train_error_GCPG.csv' (note: named "error" but stores rewards).
        - Assumes 4 cells and fixed hyperparameters for PolicyV2.
    """
    # Initialize the PolicyV2 model and set to evaluation mode on device
    model = PolicyV2()
    model.eval().to(device)


    cell_num = 4  # Fixed number of cells (nodes)
    # Initialize CPG environment with 4 cells and 500-step episode length
    env = CPGEnv(cell_nums=cell_num, env_length=500)
    # Generate edge indices for the graph and move to device
    ei = generate_edge_idx(cell_num=cell_num).to(device)


    # Get current working directory and open CSV file for writing rewards
    cwd = os.getcwd()
    csvfile = open(cwd + '/data/train_error_GCPG.csv', 'w')
    writer = csv.writer(csvfile)

    # Loop over checkpoint steps from 5k to 300k in 5k increments
    for i in np.arange(5, 305, 5):
        # Load checkpoint from file (assumes specific naming and path)
        checkpoint = torch.load(cwd + '/checkpoints/GCPG-1-512/model-' + str(i) + '0000-CPG_r_i.pt', weights_only=True) 
  
        # Load policy state dict into the model
        model.load_state_dict(checkpoint['policy_state_dict'])
        # Compute training rewards for the current checkpoint
        reward_sum_list = get_train_error(cell_num=cell_num, edge_index=ei, model=model, env=env)

        # Write rewards to CSV and print progress
        writer.writerow(reward_sum_list)

        print('steps: ', i, ' reward_sum: ', reward_sum_list)
