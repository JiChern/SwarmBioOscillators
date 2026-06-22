from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys,os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_utils import (
    load_policy,
    make_cds_env,
    make_edge_index,
    model_action,
    step_intrinsic_dynamics,
)


def get_data(cell_num, edge_index, model, env, d0, d1, length, intrinsic_dynamics):
    """
    Generate data for a given number of cells over a specified time length.
    
    Args:
        cell_num (int): Number of cells in the CDS network
        edge_index (torch.Tensor): Edge indices for the graph neural network
        model (Policy): Trained policy model
        env (CDSEnv): CDS environment instance
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
    # env.z_mat = np.random.uniform(low=-1,high=1, size=(cell_num,2))
    obs = env.z_mat.ravel()

    # Set the initial desired phase lags (reverse-mode) and construct the observation
    env.desired_lag = d0
    rl_encoding = env.encoding_angle(env.desired_lag)
    state = np.concatenate((obs,rl_encoding.ravel())) # The observation contains the node states and desired phase lags

    for i in range (length):

        # x_dp switches to traveling waves after 800 steps
        if i>800:
            env.desired_lag = d1

        action = model_action(model, state, edge_index, cell_num)
        nextstate = step_intrinsic_dynamics(env, action, intrinsic_dynamics)

        # Store x-coord values of all nodes
        if intrinsic_dynamics == 'harmonic':
            for j in range(cell_num):
                s_out[j,i] = state[2*j+1]
        else:
            for j in range(cell_num):
                s_out[j,i] = state[2*j]


        # Update the observation
        state = nextstate
    
    return s_out




if __name__ == '__main__':
    heads = 8
    fd = 64

    model = load_policy(heads=heads, feature_dim=fd)
    intrinsic_dynamics = 'hopf'

    # Determine SIES is applied to which scale of CDS.
    cell_num = 36
    ei = make_edge_index(cell_num=cell_num)
    
    # Set-up the CDS env
    env = make_cds_env(cell_nums=cell_num,env_length=500)
    

    # Determine the the x_dp specifications for different scale of network

    # x_dps for 6 units
    # env.desired_lag_list = np.array([[0,0,0,0.5,0.5,0.5],       # reverse mode 
    #                                  [0,1/6,2/6,3/6,4/6,5/6]    # traveling waves 
    #                                 ])



    # x_dps for 36 units
    # env.desired_lag_list = np.array([np.concatenate((np.zeros(18),0.5*np.ones(18))),     # reverse mode     
    #                                  np.arange(0,1,1/36)                                 # traveling waves 
    #                             ])

    reversed1 = np.zeros(int(cell_num/2))
    reversed2 = 0.5*np.ones(int(cell_num/2))
    reversed = np.concatenate([reversed1,reversed2])

    trav = np.arange(0,1,1/cell_num)

    # Execute graph-CPG and get data
    s_out = get_data(cell_num=cell_num, edge_index=ei, model=model,env=env,d0=reversed, d1=trav,length=2000, intrinsic_dynamics=intrinsic_dynamics)
    data_dir = Path.cwd() / 'data'
    data_dir.mkdir(exist_ok=True)
    np.savetxt(data_dir / f'states_{intrinsic_dynamics}_{cell_num}.csv', s_out, delimiter=",")


    for i in range(cell_num):
        plt.plot(s_out[i,:], color = [0.5,0+i*(1/cell_num),0+i*(1/cell_num)],label='x'+str(i)) 

    plt.legend()
    plt.show()
