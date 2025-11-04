import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (SCPG models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import PolicyV2 # import SCPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx, state_to_goal1


import csv

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


def get_eval_accuracy(cell_num, edge_index, model, env):
    """
    Compute evaluation accuracy (SPD values) for multiple gait targets and random seeds.

    Args:
        cell_num (int): Number of cells (nodes) in the graph.
        edge_index (torch.Tensor): Edge indices for the graph.
        model (SCPG): Trained GNN policy model.
        env (CPGEnv): CPG environment instance.

    Returns:
        numpy.ndarray: Array of SPDs for all targets and seeds.
    
    """

    # Evaluation target set
    env.desired_lag_list = np.array([[0,0,0,0],          
                                        [0,0.75,0.5,0.25],       
                                        [0,0.5,0.5,0.5],
                                        [0,0.125,0.25,0.375],
                                        ])

    SPD_list = np.array([])

    # Loop over each target gait
    for target in range(4):
        env.desired_lag = env.desired_lag_list[target]
        error_list = np.zeros(10)
        e_length = 500
        eval_step = 400

        # Run 10 episodes per target
        for s in range(10):
            state = reset(env)
            
            # Simulate 500 steps
            for p in range (500):
                # Prepare GNN input from state
                gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)
    
                # Compute action using the model
                with torch.no_grad():
                    action = model(gnn_x, edge_index)
                    action.clamp_(-1, 1)
                    action = action.squeeze().cpu().numpy()

                # Compute and store angle divergneces after sufficient steps
                if p > eval_step:
                    angle_list = state_to_goal1(state[0:cell_num*2], cell_num=cell_num)
                    angle_diff = np.zeros(cell_num)
                    for angle in range(cell_num):
                        angle_diff[angle] = abs((angle_list[angle]-env.desired_lag[angle]))
                        if angle_diff[angle] > 0.5:
                            angle_diff[angle] = 1-angle_diff[angle]
                    angle_diff_mean = np.mean(angle_diff)

                    error_list[s] += angle_diff_mean*360

                # Step the environment
                nextstate = env.step_env(action)
                state = nextstate

        # Compute and record SPD values
        error_list = error_list/(e_length-eval_step)
        SPD_list = np.concatenate((SPD_list, error_list))
        
      
    
    
    return SPD_list




if __name__ == '__main__':

    # Set-up the SCPG model
    model = PolicyV2()
    model.eval().to(device)

    cell_num = 4
    env = CPGEnv(cell_nums=cell_num,env_length=500)
    ei = generate_edge_idx(cell_num=cell_num).to(device)

    # Get current working directory and open CSV file for writing SPDs
    cwd = os.getcwd()
    csvfile = open(cwd+'/data/eval_SPD_SCPG.csv', 'w')
    writer = csv.writer(csvfile)

    # Loop over checkpoint steps from 5k to 300k in 5k increments
    for i in np.arange(300,305,5):
        checkpoint = torch.load(cwd+'/checkpoints/GCPG-1-512/model-'+str(i)+'0000-CPG_r_i.pt', weights_only=True) 
  
        model.load_state_dict(checkpoint['policy_state_dict'])
        SPD_list = get_eval_accuracy(cell_num=cell_num, edge_index=ei, model=model,env=env)


        writer.writerow(SPD_list)

        print('steps: ', i, ' SPD: ', SPD_list)



