import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (SCPG models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import Policy # import SCPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx, phase_distance, state_to_goal1
import matplotlib.pyplot as plt

import csv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Set deisred phase configuration
desired_lag = np.array([0,0.5,0,0.5]) # bound


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
    dt = 1/hz
    env = CPGEnv(cell_nums=cell_num,env_length=500,hz=hz)
    env.omega = np.pi*2
    edge_index = generate_edge_idx(cell_num).to(device)
    

    # Set-up data recorder
    cwd = os.getcwd()
    f = open(cwd+'/data/s_cpg_converge_time_bound.csv',"w")
    writer = csv.writer(f)

    # Load the initial condition shared for all CPG models
    init_conds  = np.loadtxt('data/init_cond.csv', delimiter=',', dtype=float)


    try:

        # Get converge times over 1000 random trials
        for i in range(1000):
            # Prepare initial state: concatenate oscillator positions and encoded desired lags
            z_x = np.array(init_conds[i,0:4])
            z_y = np.array(init_conds[i,4:8])
            env.z_mat = np.zeros((cell_num,2))
            env.z_mat[0,0] = z_x[0]
            env.z_mat[1,0] = z_x[1]
            env.z_mat[2,0] = z_x[2]
            env.z_mat[3,0] = z_x[3]
            env.z_mat[0,1] = z_y[0]
            env.z_mat[1,1] = z_y[1]
            env.z_mat[2,1] = z_y[2]
            env.z_mat[3,1] = z_y[3]
            obs = env.z_mat.ravel()
            env.desired_lag = desired_lag
            rl_encoding = env.encoding_angle(env.desired_lag)
            state = np.concatenate((obs,rl_encoding.ravel()))

            # initialize the converge time variable
            converge_time = 0
            
            # if converged break the loop
            done = False
            time_step = 0

            # Compute phase distance between the initial state and the "walk" state
            phase = state_to_goal1(state[0:cell_num*2], cell_num=cell_num)
            phase_dist = phase_distance(phase*2*np.pi, env.desired_lag*2*np.pi)


            while not done:  
                # Calculate phase divergence of all unit
                phase = state_to_goal1(state[0:cell_num*2], cell_num=cell_num)
                angle_diff = np.zeros(cell_num)
                for p in range(cell_num):
                    angle_diff[p] = abs((phase[p]-env.desired_lag[p]))
                    if angle_diff[p] > 0.5:
                        angle_diff[p] = 1-angle_diff[p]
                
                # Approx 5 degrees
                threshold = 0.014
                all_less_than_threshold = np.all(angle_diff < threshold)
                

                # Converge is considered when phase divergence of all unit less than 5 degrees
                if all_less_than_threshold:
                    print('converge time: ', converge_time)
                    writer.writerow([converge_time, phase_dist])
                    done = True
                
                # Simulate for 10s
                if converge_time > 10:
                    writer.writerow([converge_time, phase_dist])
                    done = True

                # Rearrange observation to GNN inputs for SCPG
                gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)

                # Obtain external coupling terms through SCPG
                with torch.no_grad():
                    action = model(gnn_x, edge_index)
                    action.clamp_(-1, 1)
                    action = action.squeeze().cpu().numpy()
                    action = action * 2

                # Execute one-step evolution of the environent
                nextstate = env.step_env(action)
                state = nextstate

                converge_time += dt
                time_step += 1

    except KeyboardInterrupt:
        print("\nScript terminated by user")


