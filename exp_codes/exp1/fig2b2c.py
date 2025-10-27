import torch
import torch_geometric
import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

from agent.networks import Policy # import graph-CPG architecture
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf,generate_edge_idx # necessary function for data adaptation to GNN-based graph-CPG model and generation of adjacency matrix
import matplotlib.pyplot as plt


def get_data(cell_num, edge_index, model, env, d0, d1, length):
    """
    Generate data for a given number of cells over a specified time length.
    
    Args:
        cell_num (int): Number of cells in the CPG network
        edge_index (torch.Tensor): Edge indices for the graph neural network
        model (Policy): Trained policy model
        env (CPGEnv): CPG environment instance
        d0 (np.ndarray): Initial desired phase lag configuration
        d1 (np.ndarray): Desired phase lag configuration after step 800
        length (int): Number of time steps to simulate
    
    Returns:
        np.ndarray: Array of state values (x-coordinates) for each cell over time
    """
    s_out = np.zeros((cell_num,length))

    # Set the initial states
    env.z_mat = np.zeros((cell_num,2))
    env.z_mat[0,0] = 0.1
    obs = env.z_mat.ravel()

    # Set the initial desired phase lags (reverse-mode) and construct the observation
    env.desired_lag = d0
    rl_encoding = env.encoding_angle(env.desired_lag)
    state = np.concatenate((obs,rl_encoding.ravel())) # The observation contains the node states and desired phase lags

    for i in range (length):

        # x_dp switches to traveling waves after 800 steps
        if i>800:
            env.desired_lag = d1

        # Rearrange observation to GNN inputs for graph-CPG
        gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num)

        # Obtain external coupling terms through graph-CPG
        with torch.no_grad():
            action = model(gnn_x, edge_index)
            action.clamp_(-1, 1)
            action = action.squeeze().cpu().numpy()
        
        # Execute one-step evolution of the environent
        nextstate = env.step_env(action)

        # Store x-coord values of all nodes
        for j in range(cell_num):
            s_out[j,i] = state[2*j]

        # Update the observation
        state = nextstate
    
    return s_out




if __name__ == '__main__':
    heads = 8
    fd = 64

    # Set-up the graph-CPG model
    model = Policy(heads=heads, feature_dim=fd)
    model.eval()
    checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])

    # Determine which scale of CDS graph-CPG is applied.
    cell_num = 6
    cell_num = 36
    ei = generate_edge_idx(cell_num=cell_num) # Generate the graph adjancency matrix of CDS
    
    # Set-up the CDS env
    env = CPGEnv(cell_nums=cell_num,env_length=500)
    

    # Determine the the x_dp specifications for different scale of network

    # x_dps for 6 units
    env.desired_lag_list = np.array([[0,0,0,0.5,0.5,0.5],       # reverse mode 
                                     [0,1/6,2/6,3/6,4/6,5/6]    # traveling waves 
                                    ])



    # x_dps for 36 units
    env.desired_lag_list = np.array([np.concatenate((np.zeros(18),0.5*np.ones(18))),     # reverse mode     
                                     np.arange(0,1,1/36)                                 # traveling waves 
                                ])

    # Execute graph-CPG and get data
    s_out = get_data(cell_num=cell_num, edge_index=ei, model=model,env=env,d0=env.desired_lag_list[0], d1=env.desired_lag_list[1],length=2000)
    data_dir = os.getcwd()+'/data/'
    np.savetxt(data_dir+'states_'+str(cell_num)+'.csv',s_out, delimiter=",")


    for i in range(cell_num):
        plt.plot(s_out[i,:], color = [0.5,0+i*(1/cell_num),0+i*(1/cell_num)],label='x'+str(i)) 

    plt.legend()
    plt.show()

