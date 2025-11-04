import torch
import torch_geometric
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import Policy # import SCPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx, state_to_goal1 
import matplotlib.pyplot as plt

import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def get_stable_loss(cell_num,model,env):
    length = 500
    edge_index = generate_edge_idx(cell_num).to(device) # Generate the graph adjancency matrix of CDS

    # Set the initial states
    env.z_mat = np.zeros((cell_num,2))
    env.z_mat[0,0] = 0.1
    obs = env.z_mat.ravel()

    # Set the desired phase lags (traveling waves) and construct the observation
    env.desired_lag = np.arange(0,1,1/cell_num)
    env.relative_lags = env.cal_relative_lags(env.desired_lag)
    rl_encoding = env.encoding_angle(env.desired_lag) # The observation contains the node states and desired phase lags
    state = np.concatenate((obs,rl_encoding.ravel()))

    # Divergence Vetor
    e_out = np.zeros(length)
    loss = 0
    
    for i in range(length):
        # Rearrange observation to GNN inputs for SCPG
        gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)

        # Obtain external coupling terms through SCPG
        with torch.no_grad():
            action = model(gnn_x, edge_index)
            action.clamp_(-1, 1)
            action = action.squeeze().cpu().numpy()
        
        # compute and record mean angle divergences
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
    
    # SPD is calculated by averaging latest 100 samples
    loss = np.mean(e_out[-100:-1])
    

    return loss




if __name__ == '__main__':

    # Set-up the SCPG model, with 8 attention heads
    heads = 8
    fd = 64
    model = Policy(heads=heads, feature_dim=fd)
    model.eval().to(device)
    checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])

    # Set-up the data recorder
    cwd = os.getcwd()
    csvfile = open(cwd+'/data/'+'SPD_traveling_waves.csv', 'w')
    writer = csv.writer(csvfile)

    for i in np.arange(2,128,1):
        cell_num = i
        # Refersh environment for different scale of network for each trial 
        env = CPGEnv(cell_nums=cell_num,env_length=500)
        loss = get_stable_loss(cell_num=cell_num,model=model,env=env)

        # Record data
        writer.writerow([i,loss])
        print('cell_num: ', i, ' loss: ', loss)

