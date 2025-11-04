import torch
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (SCPG models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import Policy # import SCPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx, state_to_goal1

import time
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_phase_data_ctrl(hz,cell_num, edge_index, model, env, target, length, Kp, Kd):

    dt = 1/hz

    # Vector recording phase divergences
    mean_out = np.zeros(length)

    
    # Prepare initial state: concatenate oscillator positions and encoded desired lags 
    env.z_mat = np.zeros((cell_num,2))
    env.z_mat[0,0] = 0.1
    obs = env.z_mat.ravel()
    env.desired_lag = target
    rl_encoding = env.encoding_angle(env.desired_lag)
    state = np.concatenate((obs,rl_encoding.ravel()))

    # Initialize control-relevant vectors
    control = np.zeros(cell_num)  # control signal
    last_error = np.zeros(cell_num) # errors of the previous control loop

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

        # Rearrange observation to GNN inputs for SCPG
        gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)

        # Obtain external coupling terms through SCPG
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
    # Set-up the SCPG model, with 8 attention heads
    heads = 8
    fd = 64
    model = Policy(heads=heads, feature_dim=fd)
    model.eval().to(device)
    checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])

    # Initialize CDS environment
    hz = 100
    cell_num = 36  # Get 72-unit results by changing 36 to 72
    env = CPGEnv(cell_nums=cell_num,env_length=500,hz=hz)
    ei = generate_edge_idx(cell_num=cell_num).to(device)


    # Set-up data recorder 
    data_dir = os.getcwd()+'/data/'
    csvfile = open(data_dir+'/rand_exp_error'+str(cell_num)+'.csv', 'w')
    writer = csv.writer(csvfile)

    # 2N trials of random x_dps
    for i in range(cell_num*2):
        rand_angles = np.random.randint(0,cell_num,cell_num)
        rand_angles[0]=0
        rand_angles = rand_angles/cell_num

        error_control = get_phase_data_ctrl(hz=hz, cell_num=cell_num, edge_index=ei, model=model,env=env,target=rand_angles,length=1000, Kp=15,Kd=0.6)  # Controlled G-CPG
        error_no_control = get_phase_data_ctrl(hz=hz, cell_num=cell_num, edge_index=ei, model=model,env=env,target=rand_angles, length=1000, Kp=0,Kd=0) # Standard G-CPG


        SPD_control = np.mean(error_control[-100:-1])
        SPD_no_control = np.mean(error_no_control[-100:-1])

        print('exp: ', i+1, ' ctrl: ', SPD_control, ' no_ctrl: ', SPD_no_control)
        
        # Record the SPD values of controlled and standard SCPG
        writer.writerow([SPD_control, SPD_no_control])

