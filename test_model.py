import torch
import torch_geometric
import sys, os
import numpy as np


from agent.networks import Policy, DoubleQFunc
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf
import matplotlib.pyplot as plt
from utils import generate_edge_idx

import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def unit_vector(vector):
    """Compute the unit vector of a given vector.
    
    Args:
        vector (np.ndarray): Input vector.
    
    Returns:
        np.ndarray: Unit vector of the input vector.
    """
    return vector / (np.linalg.norm(vector)+1e-5)

def angle_between(v1, v2):
    """Calculate the angle (in radians) between two vectors.
    
    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
    
    Returns:
        float: Angle between v1 and v2 in radians.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
def state_to_goal(state, cell_num):
    """Convert full state vector to phase relative to first oscillator for each oscillator.
    
    Args:
        state (np.ndarray): Full state vector (shape: [cell_nums * 2]).
        cell_num (int): Number of oscillators.
    
    Returns:
        np.ndarray: Phase goals for each oscillator (shape: [cell_nums]).
    """
    state_2d = np.reshape(state,(cell_num,2))
    goal_list = np.zeros(cell_num)

    for i in range(cell_num):
        z_0 = np.array([state_2d[0,0],state_2d[0,1]])
        z_i = np.array([state_2d[i,0],state_2d[i,1]])


        angle_radians =  angle_between(z_0,z_i)
        # Calculate dot product
 
        angle_sign = np.sign(np.cross(z_0,z_i))

        if angle_sign < 0:
            angle_radians = 1-(angle_radians/(2*np.pi)) 
        else:
            angle_radians = angle_radians/(2*np.pi)

        goal_list[i] = angle_radians

    return goal_list

def get_stable_loss(cell_num,model,env, x_dp):
    """Evaluate the stable loss of model by computing phase divergence from desired lags.
    
    Args:
        cell_num (int): Number of oscillators.
        model (Policy): Trained policy network (ActorNet).
        env (CPGEnv): Environment with oscillator dynamics.
        x_dp (np.ndarray): Desired phase lags (shape: [cell_nums]).
    
    Returns:
        tuple[float, np.ndarray]:
            - mean_loss (float): Mean phase divergence over the last 200 steps.
            - s_out (np.ndarray): Oscillator states over time (shape: [cell_nums, 1000]).
    """
    model.eval()
    model.to(device)

    # Coupled oscillator system is assumed to be fully connected
    edge_index = generate_edge_idx(cell_num).to(device)


    done = False
    env.z_mat = np.zeros((cell_num,2))
    env.z_mat[0,0] = 0.1
    obs = env.z_mat.ravel()

    env.desired_lag = x_dp

    env.relative_lags = env.cal_relative_lags(env.desired_lag)
    rl_encoding = env.encoding_angle(env.desired_lag)

    # Prepare initial state: concatenate oscillator positions and encoded desired lags
    state = np.concatenate((obs,rl_encoding.ravel()))
    e_out = np.zeros(1000)
    s_out = np.zeros((cell_num,1000))
    loss = 0
    
    
    for i in range(1000):
        gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)

        # Disable gradient computation (inference only) and get the action based on model and states
        with torch.no_grad():
            action = model(gnn_x, edge_index)
            action.clamp_(-1, 1)
            action = action.squeeze().cpu().numpy()
        
        max_angle = 0

        # Compute phase divergence (mean absolute difference between current and desired lags)      
        goal_list = state_to_goal(state[0:cell_num*2], cell_num=cell_num)
        angle_diff = np.zeros(cell_num)
        for p in range(cell_num):
            angle_diff[p] = abs((goal_list[p]-env.desired_lag[p]))
            if angle_diff[p] > 0.5:
                angle_diff[p] = 1-angle_diff[p]
        max_angle = np.mean(angle_diff)
        e_out[i] = max_angle*360

        # Step the environment with the computed action
        nextstate = env.step_env(action)
        state = nextstate

        # Store oscillator states (x-positions) for plotting
        for j in range(cell_num):
            s_out[j,i] = state[2*j]
    
    loss = np.mean(e_out[-200:-1])
    return loss, s_out



if __name__ == '__main__':
    # Number of attention heads and the feature space dimension used in policy
    heads = 8
    fd = 64

    # Initialize policy network
    model = Policy(state_dim=16+8, action_dim=8, heads=heads, feature_dim=fd)

    # Load model checkpoint 
    cwd = os.getcwd()
    checkpoint = torch.load(cwd+'/model_params/model-8-64.pt', weights_only=True)

    # checkpoint = torch.load('/home/jichen/test_env/scripts/train/checkpoints/multi-goal-1-512/model-2050000-CPG_r_i.pt', weights_only=True)


    model.load_state_dict(checkpoint['policy_state_dict'])

    # Initialize environment (4 coupled oscillators, 500 steps, 100Hz sampling)
    # you define arbitrary number of nodes, to test the model's generalization ability in different network scale. 
    cell_num = 16
    env = CPGEnv(cell_nums=cell_num,env_length=500,hz=100)

    # Desired phase lags (equally spaced around the unit circle)
    x_dp = np.arange(0,1,1/cell_num)

    # Optional: Randomize desired phase lags (commented out by default)
    # You can also set the x_dp to be random angles with \theta_i = p*1/N with p is a random integer between 0 to N
    # rand_angles = np.random.randint(0,cell_num,cell_num)
    # rand_angles[0]=0
    # rand_angles = rand_angles/cell_num
    # x_dp = rand_angles

    # Evaluate the model and get phase divergence and oscillator states
    loss, s_out = get_stable_loss(cell_num=cell_num,model=model,env=env, x_dp=x_dp)
    print('Desired Phase Lags: ',x_dp, ' Mean phase divergence (in Degree): ', loss)

    # Plot oscillator states over time (each line represents one oscillator)
    for i in range(cell_num):
        plt.plot(s_out[i,:], color = [0.5,0+i*(1/cell_num),0+i*(1/cell_num)], label=f'x {i}') 

    plt.legend() 
    plt.show()


