import torch
import torch_geometric
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
from numpy.random import random
from numpy import genfromtxt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def reset(env, cell_num):

    # Reset initial states randomly distrubuted on a circle (mimicking limit cycles)
    z_x = np.random.uniform(-1,1, cell_num)
    radius = 1*np.ones(cell_num)
    z_y = np.sqrt(radius*radius-z_x*z_x)
    z_y = np.where(random(cell_num) > 0.5, -z_y, z_y) 
    env.z_mat = np.array([z_x,z_y]).transpose()
    obs = env.z_mat.ravel()
    
    # Set the desired phase lags (traveling waves) and construct the observation
    env.desired_lag = np.arange(0,1,1/cell_num)
    rl_encoding = env.encoding_angle(env.desired_lag)
    obs = np.concatenate((obs,rl_encoding.ravel()))

    env.internal_step = 0

    return obs


def get_converge_step(cell_num,model,env, stable_loss):
    
    # Generate the graph adjancency matrix of CDS
    edge_index = generate_edge_idx(cell_num).to(device)

    # 100 random trials
    num_init_cond = 100

    # Convergence step vector
    converge_steps = np.zeros(num_init_cond)

    for s in range(num_init_cond):

        converge_step = 10000
        
        # Reset env with random init states and traveling wave configuration
        state = reset(env, cell_num)

        for i in range(1000):
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
            max_angle = np.mean(angle_diff)*360

            
            if max_angle*0.9 < stable_loss:
                converge_step = i
                break
            
            # Execute one-step evolution of the environent
            nextstate = env.step_env(action)
            state = nextstate
        
        # Record convergence steps after mean angle divergence reaches 110% of SPD value
        converge_steps[s] = converge_step


    return converge_steps



if __name__ == '__main__':
    # Set-up the graph-CPG model, with 8 attention heads
    heads = 8
    fd = 64
    model = Policy(heads=heads, feature_dim=fd)
    model.eval().to(device)
    checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])

    # Set-up the data writer
    cwd = os.getcwd()
    csvfile = open(cwd+'/data/'+'converge_time.csv', 'w')
    writer = csv.writer(csvfile)

    stable_loss = genfromtxt(cwd+'/data/'+'stable_loss_ref.csv', delimiter=",")

    for i in np.arange(2,121,1):  # Tested on 2 to 120 nodes

        cell_num = i
        env = CPGEnv(cell_nums=cell_num,env_length=500)
        loss = stable_loss[i-2,1]
        converge_steps = get_converge_step(cell_num=i, model=model, env=env, stable_loss=loss)

        writer.writerow(converge_steps)
        print('cell_num: ', i, 'converge_steps: ', converge_steps)


