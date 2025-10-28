import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import PolicyMLP # import V1 architecture
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
        model (PolicyStateSpace): Trained GNN policy model.
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
    reward_list = np.array([])

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
    
                # Compute action using the model
                with torch.no_grad():
                    state = torch.Tensor(state).to(device)
                    action = model(state)
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

    # Set-up the V1 model
    model = PolicyMLP(state_dim=8+8, action_dim=8, hidden_size=256)
    model.eval().to(device)


    cell_num = 4

    env = CPGEnv(cell_nums=cell_num,env_length=500)
    ei = generate_edge_idx(cell_num=cell_num).to(device)



    idx_min = 0
    e_min = 10
    # target = 0

    cwd = os.getcwd()
    csvfile = open(cwd+'/data/eval_error_V1.csv', 'w')
    writer = csv.writer(csvfile)

    for i in np.arange(5,305,5):
        checkpoint = torch.load(cwd+'/checkpoints/V1/model-'+str(i)+'0000-CPG_r_i.pt', weights_only=True) 
  
        model.load_state_dict(checkpoint['policy_state_dict'])
        reward_sum_list = get_eval_error(cell_num=cell_num, edge_index=ei, model=model,env=env)


        writer.writerow(reward_sum_list)

        print('steps: ', i, ' reward_sum: ', reward_sum_list)



