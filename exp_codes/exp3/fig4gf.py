import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (SCPG models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import Policy # import SCPG architecture
# from environment.env import CPGEnv
from environment.env_torch import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx_k, state_to_goal_torch1,rearrange_state_vector_torch

import time
import csv

import cProfile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def get_phase_data_ctrl(cell_num, edge_index, model, env, target, length, Kp, Kd):

    mean_out_ctrl = torch.zeros(100, dtype=torch.float16, device=device)
    mean_out = torch.zeros(100, dtype=torch.float16, device=device)

    # Prepare initial state: concatenate oscillator positions and encoded desired lags 
    env.z_mat = torch.zeros((cell_num, 2), dtype=torch.float16, device=device)
    env.z_mat[0,0] = 0.1
    obs = env.z_mat.ravel()
    env.desired_lag = target
    rl_encoding = env.encoding_angle(env.desired_lag).half()
    state = torch.concatenate((obs,rl_encoding.ravel())).half()

    # Initialize control-relevant vectors
    control = torch.zeros(cell_num, dtype=torch.float16, device=device)
    last_error = torch.zeros(cell_num, dtype=torch.float16, device=device)
    dt = 0.01

    env.desired_lag = target

    
    initial_angles = state_to_goal_torch1(state[0:cell_num*2], cell_num=cell_num)
    last_error = target - initial_angles  
    d_error = torch.zeros(cell_num, dtype=torch.float16, device=device)

    for i in range (length):

        # Feedback control PD control mechanism
        current_angles = state_to_goal_torch1(state[0:cell_num*2], cell_num=cell_num)

        # get the errors between the desired and the current phase lags and it's first-order derivative
        error = (target - current_angles)
        d_error = (error - last_error) / dt  

        # Compute the control signal
        control = Kp*error + Kd*d_error
        control[0] = 0

        # Control applied after 800 steps
        if i>=800:
            env.desired_lag = target + control

        # Record the current error vector for computing the derivative at next loop
        # last_error = error
        last_error.copy_(error)

        # Rearrange observation to GNN inputs for SCPG
        gnn_x = rearrange_state_vector_torch(state=state.half(), num_nodes=cell_num).half()

        # Obtain external coupling terms through SCPG
        with torch.no_grad():
            action = model(gnn_x, edge_index)
            action.clamp_(-1, 1)
            # action = action.squeeze().cpu().numpy()
            action = action.squeeze()
        
        # Sample angle divs of uncontrolled system at last 700-800 steps for calculating SPD
        if i>=(length-300) and i<length-200:
            angle_diff = torch.abs(current_angles - target)
            angle_diff = torch.where(angle_diff > 0.5, 1 - angle_diff, angle_diff)
            mean_angle = torch.mean(angle_diff)
            mean_out[i-700] = mean_angle*360

        # Sample angle divs of controlled system at last 100 steps for calculating SPD
        if i >= length - 100:
            angle_diff = torch.abs(current_angles - target)
            angle_diff = torch.where(angle_diff > 0.5, 1 - angle_diff, angle_diff)
            mean_angle = torch.mean(angle_diff)
            mean_out_ctrl[i-900] = mean_angle*360

        # print(action)

        # Execute one-step evolution of the environent
        nextstate = env.step_env(action)
        state = nextstate
        
    
    return mean_out_ctrl,mean_out

if __name__ == '__main__':
    profiler = cProfile.Profile()
    
    # Set-up the SCPG model, with 8 attention heads
    heads = 8
    fd = 64
    model = Policy(heads=heads, feature_dim=fd)
    model.eval().to(device)
    checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])
    
    model.half() # 或 model.bfloat16() 

    cell_num = 36  # set to any cell_nums as you want
    dp_num = cell_num * 2  # 2N random x_dps for the network, you can test more random desired phase lags

    # Get random desired phase-lags
    x_dp_list = torch.zeros((dp_num,cell_num)).to(device)
    x_dp_list = x_dp_list.half()
    for i in range(dp_num):
        rand_angles = torch.tensor(np.random.randint(0,cell_num,cell_num))
        rand_angles[0]=0  # First element must be 0 as each element indicates the disired phase lag between this cell and the first cell.
        rand_angles = rand_angles/cell_num
        x_dp_list[i,:] = rand_angles
    

    # Initialize CDS environment
    hz = 100
    env = CPGEnv(cell_nums=cell_num, env_length=500,hz=hz, device=device)
    env.to_half()
    # env = CPGEnv(cell_nums=cell_num,env_length=500,hz=hz)
    env.omega = torch.tensor(np.pi * 2, dtype=torch.float16, device=device)
    length = 1000
    
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

            # Initialize the SPD vectors for both controlled and standard SCPG
            SPD_list_ctrl = np.zeros(dp_num)
            SPD_list = np.zeros(dp_num)
            
            # Loop over different desired phase lags
            for idx, x_dp in enumerate(x_dp_list):

                mean_out_ctrl, mean_out = get_phase_data_ctrl(cell_num, edge_index, model, env, x_dp, length, 15, 0.6) # Controlled G-CPG

                
                SPD_ctrl = torch.mean(mean_out_ctrl)
                SPD = torch.mean(mean_out)
                print('cell_num: ', cell_num, ' neighbors: ', k , ' idx: ', idx,' SPD: ', SPD.item(), ' SPD control: ', SPD_ctrl.item())

                SPD_list_ctrl[idx] = SPD_ctrl.item()
                SPD_list[idx] = SPD.item()
            
            # Record the SPD data for a network's indegree = k
            writer_ctrl.writerow(SPD_list_ctrl)
            writer.writerow(SPD_list)
            f_ctrl.flush()
            f.flush()

    except KeyboardInterrupt:
            print("\nReceived Ctrl+C, exiting gracefully...")
            sys.exit(0)  # Exit with status code 0 (success)

 
                

