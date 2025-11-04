import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (SCPG models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

# from networks import Policy # import SCPG architecture
from agent.networks import PolicyV2 # import SCPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx, phase_distance, state_to_goal1


import csv

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
    z_x = np.random.uniform(-1,1, 4)
    radius = 1*np.ones(4)
    z_y = np.sqrt(radius*radius-z_x*z_x)
    z_y = np.where(random(4) > 0.5, -z_y, z_y) 
    env.z_mat = np.array([z_x,z_y]).transpose()
    obs = env.z_mat.ravel()
    

    rl_encoding = env.encoding_angle(env.desired_lag)
    obs = np.concatenate((obs,rl_encoding.ravel()))
    env.internal_step = 0

    return obs


def get_eval_error(cell_num, edge_index, model, env):
    """
    Compute evaluation errors (cumulative rewards) for multiple gait targets and random seeds.

    Args:
        cell_num (int): Number of cells (nodes) in the graph.
        edge_index (torch.Tensor): Edge indices for the graph.
        model (SCPG): Trained GNN policy model.
        env (CPGEnv): CPG environment instance.

    Returns:
        numpy.ndarray: Array of cumulative rewards for all targets and seeds.
    
    """
    # Evaluation target set
    env.desired_lag_list = np.array([[0,0,0,0],          
                                        [0,0.75,0.5,0.25],       
                                        [0,0.5,0.5,0.5],
                                        [0,0.125,0.25,0.375],
                                        ])

    reward_list = np.array([]) # Accumulate all rewards

    # Loop over each target gait
    for target in range(4):
        env.desired_lag = env.desired_lag_list[target]
        reward_sum_list = np.zeros(10)

        # Run 10 episodes per target
        for s in range(10):
            state = reset(env)
            
            # Simulate 250 steps
            for p in range (250):

                env.relative_lags = env.cal_relative_lags(env.desired_lag)
                # Prepare GNN input from state
                gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)

                # Compute action using the model
                with torch.no_grad():
                    action = model(gnn_x, edge_index)
                    action.clamp_(-1, 1)
                    action = action.squeeze().cpu().numpy()
                
                # Step the environment and accumulate reward
                nextstate,reward,_,_ = env.step(action)
                reward_sum_list[s] += reward
                state = nextstate
                
        # Append rewards for this target to the list
        reward_list = np.concatenate((reward_list, reward_sum_list))
        
      
    
    
    return reward_list



if __name__ == '__main__':
    """
    Main execution block: Set up model and environment, evaluate checkpoints, and save training rewards to CSV.
    
    Notes:
        - Evaluates model checkpoints from 5k to 300k steps (in 5k increments).
        - Computes rewards for predefined gait targets to generate training performance metrics.
        - Saves cumulative rewards per checkpoint to 'train_error_V0.csv' (note: named "error" but stores rewards).
        - Assumes 4 cells and fixed hyperparameters (heads=1, feature_dim=512) for PolicyStateSpace.
        - Prints model state_dict keys for verification.
    """

    # Set-up the SCPG model
    model = PolicyV2()
    model.eval().to(device)

    cell_num = 4
    env = CPGEnv(cell_nums=cell_num,env_length=500)
    ei = generate_edge_idx(cell_num=cell_num).to(device)

    # Get current working directory and open CSV file for writing rewards
    cwd = os.getcwd()
    csvfile = open(cwd+'/data/eval_error_SCPG.csv', 'w')
    writer = csv.writer(csvfile)

    # Loop over checkpoint steps from 5k to 300k in 5k increments
    for i in np.arange(5,305,5):
        checkpoint = torch.load(cwd+'/checkpoints/GCPG-1-512/model-'+str(i)+'0000-CPG_r_i.pt', weights_only=True) 
  
        model.load_state_dict(checkpoint['policy_state_dict'])
        reward_sum_list = get_eval_error(cell_num=cell_num, edge_index=ei, model=model,env=env)


        writer.writerow(reward_sum_list)

        print('steps: ', i, ' reward_sum: ', reward_sum_list)



