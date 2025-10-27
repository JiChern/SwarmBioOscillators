import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import Policy # import graph-CPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx_k, state_to_goal1

import time
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def get_phase_data_ctrl(cell_num, edge_index, model, env, target, length, Kp, Kd):

    mean_out = np.zeros(length)

    # Prepare initial state: concatenate oscillator positions and encoded desired lags 
    env.z_mat = np.zeros((cell_num,2))
    env.z_mat[0,0] = 0.1
    obs = env.z_mat.ravel()
    env.desired_lag = target
    rl_encoding = env.encoding_angle(env.desired_lag)
    state = np.concatenate((obs,rl_encoding.ravel()))

    # Initialize control-relevant vectors
    control = np.zeros(cell_num) # control signal
    last_error = np.zeros(cell_num) # errors of the previous control loop
    dt = 0.01


    for i in range (length):

        # Feedback control PD control mechanism
        current_angles = state_to_goal1(state[0:cell_num*2], cell_num=cell_num)

        # get the errors between the desired and the current phase lags and it's first-order derivative
        error = (target - current_angles)
        if i>1:
            d_error = (error - last_error)/dt
        else:
            d_error = np.zeros(cell_num)

        # Compute the control signal
        control = Kp*error + Kd*d_error
        control[0] = 0

        # Control applied after 800 steps
        if i<800:
            env.desired_lag = target
        else:
            env.desired_lag = target + control

        # Record the current error vector for computing the derivative at next loop
        last_error = error

        # Rearrange observation to GNN inputs for graph-CPG
        gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)

        # Obtain external coupling terms through graph-CPG
        with torch.no_grad():
            action = model(gnn_x, edge_index)
            action.clamp_(-1, 1)
            action = action.squeeze().cpu().numpy()
        
        # Calculate mean phase divergence of all unit
        angle_diff = np.zeros(cell_num)
        for p in range(cell_num):
            angle_diff[p] = abs((current_angles[p]-target[p]))
            if angle_diff[p] > 0.5:
                angle_diff[p] = 1-angle_diff[p]

        mean_angle = np.mean(angle_diff)
        mean_out[i] = mean_angle*360

        # Execute one-step evolution of the environent
        nextstate = env.step_env(action)
        state = nextstate
        
    
    return mean_out

if __name__ == '__main__':

    # Set-up the graph-CPG model, with 8 attention heads
    heads = 8
    fd = 64
    model = Policy(heads=heads, feature_dim=fd)
    model.eval().to(device)
    checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])
    
    
    cell_num = 18  # set to any cell_nums as you want
    dp_num = 50  # 50 random x_dps for the network, you can test more random desired phase lags

    # Get random desired phase-lags
    x_dp_list = np.zeros((dp_num,cell_num))
    for i in range(dp_num):
        rand_angles = np.random.randint(0,cell_num,cell_num)
        rand_angles[0]=0  # First element must be 0 as each element indicates the disired phase lag between this cell and the first cell.
        rand_angles = rand_angles/cell_num
        x_dp_list[i,:] = rand_angles
    

    # Initialize CDS environment
    hz = 100
    env = CPGEnv(cell_nums=cell_num,env_length=500,hz=hz)
    env.omega = np.pi*2
    length = 1200
    
    # Set-up data recorder 
    data_dir = os.getcwd()+'/data/'
    f = open(data_dir+'/sparsity'+str(cell_num)+'.csv', 'w')
    writer = csv.writer(f)
    f_ctrl = open(data_dir+'/sparsity_ctrl'+str(cell_num)+'.csv', 'w')
    writer_ctrl = csv.writer(f_ctrl)

    try:
        # Loop over different in-degrees of the networks
        for k in np.arange(1,cell_num,1):
            edge_index = generate_edge_idx_k(cell_num, k).to(device) # this function generate the graph-adjancy matrix for sparse networks based on the nearest-neighboring rule.

            # Initialize the SPD vectors for both controlled and standard graph-CPG
            SPD_list_ctrl = np.zeros(dp_num)
            SPD_list = np.zeros(dp_num)
            
            # Loop over different desired phase lags
            for idx, x_dp in enumerate(x_dp_list):

                mean_out_ctrl = get_phase_data_ctrl(cell_num, edge_index, model, env, x_dp, length, 15, 0.6) # Controlled G-CPG
                mean_out = get_phase_data_ctrl(cell_num, edge_index, model, env, x_dp, length, 0, 0) # Standard G-CPG

                
                SPD_ctrl = np.mean(mean_out_ctrl[-100:-1])
                SPD = np.mean(mean_out[-100:-1])
                print('cell_num: ', cell_num, ' neighbors: ', k , ' idx: ', idx,' SPD: ', SPD, ' SPD control: ', SPD_ctrl)

                SPD_list_ctrl[idx] = SPD_ctrl
                SPD_list[idx] = SPD
            
            # Record the SPD data for a network's indegree = k
            writer_ctrl.writerow(SPD_list_ctrl)
            writer.writerow(SPD_list)

    except KeyboardInterrupt:
            print("\nReceived Ctrl+C, exiting gracefully...")
            sys.exit(0)  # Exit with status code 0 (success)

                
                

