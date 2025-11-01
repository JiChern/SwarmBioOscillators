import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import Policy # import graph-CPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx, state_to_goal1 
import matplotlib.pyplot as plt

import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def get_phase_data(cell_num, edge_index, model, env, d0, length):

    e_out = np.zeros(length)

    # Set the initial states
    env.z_mat = np.zeros((cell_num,2))
    env.z_mat[0,0] = 0.1
    obs = env.z_mat.ravel()

    # Set the desired phase lags and construct the observation
    env.desired_lag = d0
    rl_encoding = env.encoding_angle(env.desired_lag)
    state = np.concatenate((obs,rl_encoding.ravel()))

    for i in range (length):
        # Rearrange observation to GNN inputs for graph-CPG
        gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)

        # Obtain external coupling terms through graph-CPG
        with torch.no_grad():
            action = model(gnn_x, edge_index)
            action.clamp_(-1, 1)
            action = action.squeeze().cpu().numpy()
        
        # Compute and mean angle divergences
        max_angle = 0
        goal_list = state_to_goal1(state[0:cell_num*2], cell_num=cell_num)
        angle_diff = np.zeros(cell_num)
        for p in range(cell_num):
            angle_diff[p] = abs((goal_list[p]-env.desired_lag[p]))
            if angle_diff[p] > 0.5:
                angle_diff[p] = 1-angle_diff[p]
        
        max_angle = np.mean(angle_diff)
        e_out[i] = max_angle*360

        # Execute one-step evolution of the environent
        nextstate = env.step_env(action)
        state = nextstate
    
    return e_out




if __name__ == '__main__':
    # Set-up the graph-CPG model, with 8 attention heads
    heads = 8
    fd = 64
    model = Policy(heads=heads, feature_dim=fd)
    model.eval().to(device)
    checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])

    # Set-up the data writer
    data_dir = os.getcwd()+'/data/'
    csvfile = open(data_dir+'/rand_angle_error.csv', 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['cell_num', 'avg_divergence'])  # Add header for clarity

    for i in range(117):
        cell_num = i+4
        env = CPGEnv(cell_nums=cell_num,env_length=500)
        ei = generate_edge_idx(cell_num).to(device)
        
        # Evaluated 2 x cell num random targets, as larger networks consists more symmetries and information
        num_experiments = 2 * cell_num

        for exp in range(num_experiments):
            # Set random desired phase lags
            rand_angles = np.random.randint(0,cell_num,cell_num)
            rand_angles[0]=0
            rand_angles = rand_angles/cell_num

            e_out = get_phase_data(cell_num=cell_num, edge_index=ei, model=model,env=env,d0=rand_angles,length=800)
            e_avg = np.mean(e_out[-100:])  # Corrected to include the last element (last 100 steps)
            print('cell_num: ', cell_num, ' exp: ', exp, ' angle_div: ',e_avg)

            # Write each individual result as a row to make the data flat and handle variable lengths
            writer.writerow([cell_num, e_avg])

    csvfile.close()
