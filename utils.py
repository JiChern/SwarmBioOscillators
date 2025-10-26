import itertools
import os
import random
from collections import deque, namedtuple

import numpy as np
import torch

from numpy.random import default_rng




def make_checkpoint(agent, step_count, env_name, dir_name):
    """ Make model checkpoint during the training """
    q_funcs, target_q_funcs, policy = agent.q_funcs, agent.target_q_funcs, agent.policy
    
    save_path = "checkpoints/"+str(dir_name)+"/model-{}-{}.pt".format(step_count, env_name)

    if not os.path.isdir("checkpoints/"+str(dir_name)):
        os.makedirs("checkpoints/"+str(dir_name))

    torch.save({
        'double_q_state_dict': q_funcs.state_dict(),
        'target_double_q_state_dict': target_q_funcs.state_dict(),
        'policy_state_dict': policy.state_dict(),
        # 'log_alpha_state_dict': log_alpha
    }, save_path)


def rearrange_state_vector_hopf(state, num_nodes):
    """ Rearrange the state vector to input the matrix of graph neural network. """
    state_dim = 2
    d_phase = torch.FloatTensor(state[state_dim*num_nodes:]).unsqueeze(dim=1).view(num_nodes,-1)
    states = torch.FloatTensor(state[0:state_dim*num_nodes]).unsqueeze(dim=0).view(num_nodes,state_dim)
    out = torch.concatenate((states,d_phase),dim=1)


    return out

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'."""
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def state_to_goal(state):
    """ Returns the phase lag for each cell relative to first cell in cycle based on the system states'."""
    
    state_2d = np.reshape(state,(4,2))
    goal_list = np.zeros(4)

    for i in range(4):
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

def generate_edge_idx(cell_num):
    """ Returns the adjacency matrix of complete graph with node number equal to cell_num."""
    
    ei = torch.zeros(2,cell_num*(cell_num-1), dtype=torch.long)
    for i in range(cell_num):
        row1 = i*torch.ones(cell_num-1)
        # print(row1)
        row2 = torch.arange(cell_num)
        row2 = row2[row2!=i]
        start_col = i*(cell_num-1)
        ei[0,start_col:start_col+cell_num-1] = row1
        ei[1,start_col:start_col+cell_num-1]  = row2


    return ei.to(device='cpu')

def generate_edge_idx_synaptic(cell_num):
    """ Returns the adjacency matrix of graph with synaptic coupling with node number equal to cell_num."""
    ei = torch.zeros(2,cell_num*2, dtype=torch.long)
    row_1_1 = torch.arange(0,cell_num,1)
    row_1_2 = torch.arange(0,cell_num,1)

    row_2_1 = torch.zeros(1,cell_num, dtype=torch.long)
    row_2_2 = torch.zeros(1,cell_num, dtype=torch.long)
    for i in range(cell_num):
        if i%2 == 0:
            ei[1,i] = i+1
        else:
            ei[1,i] = i-1

        ei[0,i] =  i
        ei[0,i+cell_num] =  i
        
        ei[1,i+cell_num] =  (i+2)%cell_num


    return ei.to(device='cpu')

