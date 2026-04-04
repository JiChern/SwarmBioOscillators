import itertools
import os
import random
from collections import deque, namedtuple
import networkx as nx
import numpy as np
import torch

from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score  # Added for ROC-AUC calculation
import random


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            rocauc_list.append(score)
    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')
    return sum(rocauc_list)/len(rocauc_list)

def set_seed(seed):
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True) 

    os.environ['PYTHONHASHSEED'] = str(seed)




def create_optimizer(model, lr, wd=1e-4):
    """ Create optimizer for CDS GNNs """
    decay = []
    no_decay = []
    special_params = [] 
    dyn_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(x in n.lower() for x in ['ks']):   # 同时捕捉 ks 和 prelu
            special_params.append(p)
        elif any(key in n.lower() for key in ['omega', 'zeta']):
            dyn_params.append(p)
        elif any(key in n.lower() for key in ['bias', 'norm']):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.Adam([
        {'params': decay,          'lr': lr,      'weight_decay': wd},      # weights （with decay）
        {'params': no_decay,       'lr': lr,      'weight_decay': 0.0},     # bias/norm（no decay）
        {'params': special_params, 'lr': lr * 10, 'weight_decay': 0.0},      # if grad enabled Ks should have larger lr and no decay
        {'params': dyn_params,     'lr': lr / 5,  'weight_decay': 0.0},       # # if grad enabled omega and zeta should have smaller lr and no decay
    ])

    print("=== Optimizer Groups ===")
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}: lr={group['lr']:.2e}, wd={group['weight_decay']}, params={len(group['params'])}")

    return optimizer


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

def rearrange_state_vector_torch(state, num_nodes):
    """Rearrange the state vector to input the matrix of graph neural network."""
    state_dim = 2
    d_phase = state[state_dim * num_nodes:].view(num_nodes, -1)
    states = state[0 : state_dim * num_nodes].view(num_nodes, state_dim)
    out = torch.cat((states, d_phase), dim=1)
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

def state_to_goal1(state, cell_num):
    """Convert full state vector to phase relative to first oscillator for each oscillator.
    
    Args:
        state (np.ndarray): Full state vector (shape: [cell_num * 2]).
        cell_num (int): Number of oscillators.
    
    Returns:
        np.ndarray: Phase goals for each oscillator (shape: [cell_num]).
    """
    # Reshape state to 2D array (cell_num, 2)
    state_2d = np.reshape(state, (cell_num, 2))
    
    # Reference vector (first oscillator)
    z_0 = state_2d[0]  # Shape: (2,)
    
    # Compute dot products and magnitudes for all oscillators
    dot_products = np.dot(state_2d, z_0)  # Shape: (cell_num,)
    norm_z_0 = np.linalg.norm(z_0)
    norm_z_i = np.linalg.norm(state_2d, axis=1)  # Shape: (cell_num,)
    
    # Compute cosines of angles (avoiding division by zero)
    cos_angles = dot_products / (norm_z_0 * norm_z_i + 1e-10)  # Add small epsilon for stability
    cos_angles = np.clip(cos_angles, -1.0, 1.0)  # Ensure valid range for arccos
    angles = np.arccos(cos_angles)  # Shape: (cell_num,)
    
    # Compute sign using cross product (vectorized)
    cross_products = z_0[0] * state_2d[:, 1] - z_0[1] * state_2d[:, 0]  # Shape: (cell_num,)
    angle_signs = np.sign(cross_products)
    
    # Adjust angles based on sign
    phase = np.where(angle_signs < 0, 1 - angles / (2 * np.pi), angles / (2 * np.pi))
    
    return phase

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

def generate_edge_idx_k(cell_num, k=2):
    """
    Generate the edge index for a graph where each node connects to k neighbors
    in a cyclic manner (e.g., node i connects to nodes i+1, i+2, ..., i+k modulo cell_num).
    
    Args:
        cell_num (int): Number of nodes in the graph.
        k (int): Number of neighbors per node (default: 2).
        
    Returns:
        torch.Tensor: Edge index tensor of shape (2, cell_num * k) with dtype torch.long.
    """
    # Ensure k is valid
    if k >= cell_num:
        raise ValueError("k must be less than cell_num to avoid a complete graph")
    
    # Initialize edge index tensor: 2 rows, cell_num * k columns for directed edges
    ei = torch.zeros(2, cell_num * k, dtype=torch.long)
    
    for i in range(cell_num):
        # Define the k neighbors for node i (cyclically: i+1, i+2, ..., i+k)
        neighbors = [(i + j + 1) % cell_num for j in range(k)]
        start_col = i * k
        # Source nodes (row 0): repeat node i for all its neighbors
        ei[0, start_col:start_col + k] = i
        # Target nodes (row 1): assign the k neighbors
        ei[1, start_col:start_col + k] = torch.tensor(neighbors, dtype=torch.long)
    
    return ei.to(device='cpu')

def phase_distance(phi, theta):
    """
    calculate the distance between two phase vector, using unit-circle embedding
    
    Args:
    phi, theta : np.ndarray
        Input phase vectors (N,)，单位：弧度。
    
    Returns:
    L : float
        distance between two phase vector using unit-circle embedding
    """
    if phi.shape != theta.shape:
        raise ValueError("phi and theta must be in equal length")
    
    # get phase difference
    diff = phi - theta
    
    # calculate the chord length: 2 - 2 cos(diff)
    dist_sq = 2 - 2 * np.cos(diff)
    
    L = np.sqrt(np.sum(dist_sq))
    
    return L

def state_to_phase_diff(state, cell_num):
    """Convert full state vector to phase relative to first oscillator for each oscillator.
    
    Args:
        state (np.ndarray): Full state vector (shape: [cell_nums * 2]).
        cell_num (int): Number of oscillators.
    
    Returns:
        np.ndarray: Phase goals for each oscillator (shape: [cell_nums]).
    """
    state_2d = np.reshape(state,(cell_num,2))
    phase_diff = np.zeros(cell_num)

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

        phase_diff[i] = angle_radians

    return phase_diff

def state_to_goal_torch(state, cell_num):
    """Convert full state vector to phase relative to first oscillator for each oscillator.
   
    Args:
        state (torch.Tensor): Full state vector (shape: [cell_num * 2]).
        cell_num (int): Number of oscillators.
   
    Returns:
        torch.Tensor: Phase goals for each oscillator (shape: [cell_num]).
    """
    # Reshape state to 2D tensor (cell_num, 2)
    state_2d = state.view(cell_num, 2)
   
    # Reference vector (first oscillator)
    z_0 = state_2d[0]  # Shape: (2,)
   
    # Compute dot products and magnitudes for all oscillators
    dot_products = torch.matmul(state_2d, z_0)  # Shape: (cell_num,)
    norm_z_0 = torch.norm(z_0)
    norm_z_i = torch.norm(state_2d, dim=1)  # Shape: (cell_num,)
   
    # Compute cosines of angles (avoiding division by zero)
    cos_angles = dot_products / (norm_z_0 * norm_z_i + 1e-10)  # Add small epsilon for stability
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)  # Ensure valid range for acos
    angles = torch.acos(cos_angles)  # Shape: (cell_num,)
   
    # Compute sign using cross product (vectorized)
    cross_products = z_0[0] * state_2d[:, 1] - z_0[1] * state_2d[:, 0]  # Shape: (cell_num,)
    angle_signs = torch.sign(cross_products)
   
    # Adjust angles based on sign
    phase = torch.where(angle_signs < 0, 1 - angles / (2 * np.pi), angles / (2 * np.pi))
   
    return phase

def state_to_goal_torch1(state, cell_num):
    """Convert full state vector to phase relative to first oscillator for each oscillator.
 
    Args:
        state (torch.Tensor): State vector. Supports batched input [batch_size, cell_num * 2] or single [cell_num * 2].
        cell_num (int): Number of oscillators.
 
    Returns:
        torch.Tensor: Phase goals for each oscillator (shape: [batch_size, cell_num] or [cell_num]).
    """
    if state.dim() == 1:
        state = state.unsqueeze(0)  # Treat as batch_size=1
    batch_size = state.shape[0]
    state_2d = state.reshape(batch_size, cell_num, 2)
    phases = torch.atan2(state_2d[:, :, 1], state_2d[:, :, 0]) / (2 * torch.pi)
    phase_0 = phases[:, 0]  # [batch_size]
    relative_phases = (phases - phase_0.unsqueeze(1)) % 1.0  # [batch_size, cell_num]
    if batch_size == 1:
        relative_phases = relative_phases.squeeze(0)
    return relative_phases


def generate_small_world_edge_index(cell_num, k=4, p=0.05, seed=42):  # 新增 seed 参数，默认固定值
    random.seed(seed)  # 设置种子，确保可复现
    G = nx.watts_strogatz_graph(n=cell_num, k=k, p=p)
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)  # 转为无向
    return edge_index

def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

if __name__ == '__main__':

    ei = generate_edge_idx_k(120,k=20)
    np.savetxt('ei.csv', ei, delimiter=',', fmt='%d')
    print(ei)

