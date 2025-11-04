import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (SCPG models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import Policy # import SCPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx 
import matplotlib.pyplot as plt

import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set deisred phase configurations for gait transition
trot = np.array([0,0.5,0.5,0])
walk = np.array([0,0.5,0.75,0.25])

if __name__ == '__main__':

    # Set-up the SCPG model, with 8 attention heads
    heads = 8
    fd = 64
    model = Policy(heads=heads, feature_dim=fd)
    model.eval().to(device)
    checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])

    # Initialize CDS environment
    cell_num = 4
    hz = 100
    env = CPGEnv(cell_nums=cell_num,env_length=500,hz=hz)
    env.omega = np.pi*2
    edge_index = generate_edge_idx(cell_num).to(device)

    # Prepare initial state: concatenate oscillator positions and encoded desired lags (trot)
    env.desired_lag = trot 
    env.z_mat = np.zeros((cell_num,2))
    env.z_mat[1,0] = 1
    env.z_mat[2,0] = 1
    env.z_mat[3,0] = 1
    obs = env.z_mat.ravel()
    env.relative_lags = env.cal_relative_lags(env.desired_lag)
    rl_encoding = env.encoding_angle(env.desired_lag)
    state = np.concatenate((obs,rl_encoding.ravel()))


    x_mat = np.array([env.z_mat[0][0], env.z_mat[1][0], env.z_mat[2][0], env.z_mat[3][0]]).reshape(-1, cell_num)

    # Set-up data recorder
    cwd = os.getcwd()
    f = open(cwd+'/data/scpg_trot_walk.csv',"w")
    writer = csv.writer(f)

    # Set start time and time-duratio vector
    dt = 1/hz
    duration = 0
    duration_vec = [0]


    try:
        while True:
            
            # Execute trot-to-walk gait transition
            if duration>5:
                env.desired_lag = walk  

            # Rearrange observation to GNN inputs for SCPG
            gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)

            # Obtain external coupling terms through SCPG
            with torch.no_grad():
                action = model(gnn_x, edge_index)
                action.clamp_(-1, 1)
                action = action.squeeze().cpu().numpy()
        
            # Execute one-step evolution of the environent
            nextstate = env.step_env(action)
            state = nextstate

            # Record state vectors for plotting
            x_vec = np.array([env.z_mat[0][0], env.z_mat[1][0], env.z_mat[2][0], env.z_mat[3][0]]).reshape(-1, cell_num)
            x_mat = np.concatenate((x_mat,x_vec),axis=0)

            # Record state vectors and duration
            writer.writerow([duration,env.z_mat[0][0], env.z_mat[1][0], env.z_mat[2][0], env.z_mat[3][0]])

            duration += dt
            duration_vec.append(duration)
            
            if duration > 10:
                break
    
    except KeyboardInterrupt:
        print('stopped')


    plt.plot(duration_vec,x_mat[:,0], label=f'x {1}')
    plt.plot(duration_vec,x_mat[:,1], label=f'x {2}')
    plt.plot(duration_vec,x_mat[:,2], label=f'x {3}')
    plt.plot(duration_vec,x_mat[:,3], label=f'x {4}')

    plt.legend()

    plt.show()
