import torch
import sys, os
import numpy as np

from pathlib import Path  # 
parent_dir = str(Path(__file__).parent.parent)  
sys.path.append(parent_dir)  

# from networks import Policy # import graph-CPG architecture
from agent.networks import Policy  # Import graph-CPG architecture (Policy is a GNN model for action computation)
from environment.env import CPGEnv  # Import CPG environment for simulating oscillatory dynamics
from utils import rearrange_state_vector_hopf, generate_edge_idx

import csv  # Import for writing results to CSV files
from numpy.random import random  # Import for random number generation


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reset env's initial states for testing
def reset(env):
    """
    Reset the CPG environment to a random initial state.

    Args:
        env (CPGEnv): The CPG environment instance.

    Returns:
        numpy.ndarray: Initial observation vector, combining states and desired phase encodings.
    
    Notes:
        - Initializes states on a unit circle for oscillatory stability.
        - Concatenates flattened states with phase encodings for RL-compatible observation.
    """
    z_x = np.random.uniform(-1, 1, 8)  # Sample random x-coordinates for 8 cells
    radius = 1 * np.ones(8)  # Unit radius for circle
    # Compute y-coordinates to lie on the unit circle, with random sign flip
    z_y = np.sqrt(radius * radius - z_x * z_x)
    z_y = np.where(random(8) > 0.5, -z_y, z_y)
    env.z_mat = np.array([z_x, z_y]).transpose()  # Set environment state matrix
    obs = env.z_mat.ravel()  # Flatten state matrix
    
    # Encode desired phase lags and concatenate to form full observation
    rl_encoding = env.encoding_angle(env.desired_lag)
    obs = np.concatenate((obs, rl_encoding.ravel()))
    env.internal_step = 0  # Reset internal step counter

    return obs


def get_train_reward(cell_num, edge_index, model, env):
    """
    Compute training errors (cumulative rewards) for multiple targets and random seeds.

    Args:
        cell_num (int): Number of cells (nodes) in the graph.
        edge_index (torch.Tensor): Edge indices for the graph.
        model (Policy): Trained GNN policy model.
        env (CPGEnv): CPG environment instance.

    Returns:
        numpy.ndarray: Array of cumulative rewards for all targets and seeds.
    
    Notes:
        - Evaluates the model on 4 predefined targets (phase lag patterns).
        - For each target, runs 10 episodes with random initial states, each 250 steps long.
        - Rewards measure how well the generated phases match the target.
    """
    # Define training target phase lag patterns (4 different patterns for 8 cells)
    env.desired_lag_list = np.array([[0, 0.5, 0.25, 0.75, 0.5, 0, 0.75, 0.25],
                                     [0, 0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25],
                                     [0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0],
                                     [0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5]])

    reward_list = np.array([])  # Accumulate all rewards

    # Loop over each target pattern
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
                reward_sum_list[s] += reward
                state = nextstate  # Update state

        # Append rewards for this target to the list
        reward_list = np.concatenate((reward_list, reward_sum_list))
        
    return reward_list


if __name__ == '__main__':
    """
    Main execution block: Set up model and environment, evaluate checkpoints, and save training errors to CSV.
    
    Notes:
        - Evaluates model checkpoints from 300k to 495k steps (in 5k increments).
        - Computes errors for predefined targets.
        - Saves cumulative rewards per checkpoint to 'train_reward_GCPG.csv'.
        - Assumes 8 cells and fixed hyperparameters (heads=8, feature_dim=64).
    """
    # Initialize GNN policy model with 8 attention heads and 64 feature dimensions
    heads = 8
    fd = 64
    model = Policy(heads=heads, feature_dim=fd)
    model.eval().to(device)  # Set model to evaluation mode on the device

    cell_num = 8  # Fixed number of cells (nodes)

    # Initialize CPG environment with 8 cells and 500-step episode length
    env = CPGEnv(cell_nums=cell_num, env_length=500)
    # Generate edge indices for the graph and move to device
    ei = generate_edge_idx(cell_num=cell_num).to(device)

    # Unused variables (potentially for tracking minimum error; not used in loop)
    idx_min = 0
    e_min = 10

    # Get current working directory and open CSV file for writing rewards
    cwd = os.getcwd()
    csvfile = open(cwd + '/data/eval_reward_GCPG.csv', 'w')
    writer = csv.writer(csvfile)

    # Loop over checkpoint steps from 50k to 305k in 5k increments
    # Note: Commented line suggests previous range (5k to 300k); current is 300k to 495k
    for i in np.arange(5,305,5):
        # Load checkpoint from file (assumes specific naming convention)
        checkpoint = torch.load(parent_dir + '/checkpoints/multi-goal-8-64/model-' + str(i) + '0000-CPG_r_i.pt', weights_only=True)
  
        # Load policy state dict into the model
        model.load_state_dict(checkpoint['policy_state_dict'])
        # Compute training errors (rewards) for the current checkpoint
        reward_sum_list = get_train_reward(cell_num=cell_num, edge_index=ei, model=model, env=env)

        # Write rewards to CSV and print progress
        writer.writerow(reward_sum_list)
        print('steps: ', i, ' reward_sum: ', reward_sum_list)