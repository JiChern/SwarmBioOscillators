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

    # Initialize the matrix stores the phase lags used for plotting
    angles = np.array([])

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

        if i > length - 101:
            angles = np.concatenate((angles, current_angles))


    

    
    return mean_out, angles




if __name__ == '__main__':
    # Set-up the graph-CPG model, with 8 attention heads
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

    traveling_waves = np.arange(0,1,1/cell_num)

    # Get the phase lag lists of both variant of Graph-CPG
    error_control, angles_ctrl = get_phase_data_ctrl(hz=hz, cell_num=cell_num, edge_index=ei, model=model,env=env,target=traveling_waves,length=1200, Kp=15,Kd=0.6)  # Controlled G-CPG
    error_no_control, angles_orig = get_phase_data_ctrl(hz=hz, cell_num=cell_num, edge_index=ei, model=model,env=env,target=traveling_waves, length=1200, Kp=0,Kd=0) # Standard G-CPG


    # Define the reference phase lags for better comparison
    reference = np.zeros(36)
    for i in range(36):
        reference[i] = i / 36

    reference = reference * 2* np.pi


    # Calculate the mean phase lags of the latest 100 steps
    angles_ctrl = angles_ctrl * 2 * np.pi
    angles_orig = angles_orig * 2 * np.pi
    angles_ctrl = angles_ctrl.reshape(-1,36)
    angles_orig = angles_orig.reshape(-1,36)
    angles_ctrl_mean =  angles_ctrl.mean(axis=0)
    angles_orig_mean = angles_orig.mean(axis=0)


    # Figure 1 - controlled graph-CPG results
    plt.figure(1)
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    plt.plot(x,y,'k','-')

    plt.plot(np.cos(reference[0]), np.sin(reference[0]), '.')
    for i in range(36):
        plt.plot([0, np.cos(reference[i])], [0, np.sin(reference[i])], color=[0.2, 0.5, 0.9, 0.2], linewidth=5)

    for i in range(36):
        plt.plot([0, np.cos(angles_ctrl_mean[i])], [0, np.sin(angles_ctrl_mean[i])], color=[0.2, 0.5, 0.9, 0.9], linewidth=2)

    for i in range(36):
        plt.plot(np.cos(angles_ctrl_mean[i]), np.sin(angles_ctrl_mean[i]), 'o', color=[0.2, 0.5, 0.9, 0.9], linewidth=2)


    fig = plt.gcf()
    fig.set_size_inches(500 / 96, 500 / 96)  # Approximate pixel to inch conversion assuming 96 dpi
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    ax.set_aspect('equal')


    # Figure 2 - standard graph-CPG results
    plt.figure(2)
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')


    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    plt.plot(x,y,'k')

    plt.plot(np.cos(reference[0]), np.sin(reference[0]), '.')
    for i in range(36):
        plt.plot([0, np.cos(reference[i])], [0, np.sin(reference[i])], color=[0.2, 0.5, 0.9, 0.2], linewidth=5)

    for i in range(36):
        plt.plot([0, np.cos(angles_orig_mean[i])], [0, np.sin(angles_orig_mean[i])], color=[0.9, 0.5, 0.5, 0.9], linewidth=2)

    for i in range(36):
        plt.plot(np.cos(angles_orig_mean[i]), np.sin(angles_orig_mean[i]), 'o', color=[0.9, 0.5, 0.5, 0.9], linewidth=2)




    fig = plt.gcf()
    fig.set_size_inches(500 / 96, 500 / 96)  # Approximate pixel to inch conversion assuming 96 dpi
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    ax.set_aspect('equal')


    plt.show()
