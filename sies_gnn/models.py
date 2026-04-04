import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basic_gnn import GATConv,GATatt
from att_block import *
from torch_geometric.utils import degree 







class Norm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm 



class SIES_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, nheads=8, dt=0.01, zeta=1.0, 
                 omega=1., Ks=1, e_dropout=0, learnable_dyn_params=False):

        super(SIES_GNN, self).__init__()
        
        self.dropout = dropout # Input and output dropout
        self.nhid = nhid

        # ODE Hyperparameters
        self.nlayers = nlayers
        self.dt = dt
        self.Ks_raw = nn.Parameter(torch.tensor(Ks), requires_grad=learnable_dyn_params)  
        self.omega_raw = nn.Parameter(torch.tensor(omega), requires_grad=learnable_dyn_params)
        self.zeta_raw = nn.Parameter(torch.tensor(zeta), requires_grad=learnable_dyn_params)

        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)

        out_dim = nhid * nheads
        self.linear_out = nn.Linear(out_dim, nhid) 
        self.act_fn = nn.ReLU() 

        # Compute the attraction and repulsive forces via signed attention
        self.coupling_block = SIES_coupling_block(
            in_channels=nhid, hidden_channels=nhid, heads=nheads,
            share_weights=False, concat=True, dropout=e_dropout
        )


        self.norm_X = Norm(nhid, eps=1e-5)
        self.norm_Y = Norm(nhid, eps=1e-5)

        self.reset_params()
        
    def reset_params(self):
        self.coupling_block.reset_parameters()
        self.linear_out.reset_parameters()
        self.enc.reset_parameters()
        self.dec.reset_parameters()

    def forward(self, x, edge_index, return_trajectories=False):
        Y = self.act_fn(self.enc(x))
        # Input dropouts
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = Y
        Y_0 = Y.clone()

        # Construct desired phase lag representation via node features
        x_dp_l_fixed = self.coupling_block.lin_dp(Y_0).view(-1, self.coupling_block.heads, self.coupling_block.hidden_channels)
        x_dp_r_fixed = self.coupling_block.lin_dp_r(Y_0).view(-1, self.coupling_block.heads, self.coupling_block.hidden_channels)

        # As att_block computes raw attention coeffs without softmax normalization, message is nornalized based on in and out degrees (similar to GCN) norm = 1/sqrt(in_degs*out_degs) 
        deg = degree(edge_index[1], num_nodes=x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        row, col = edge_index
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        omega = F.softplus(self.omega_raw)   
        zeta  = F.softplus(self.zeta_raw)
        Ks    = F.softplus(self.Ks_raw)
        omega_square = omega ** 2 

        # Integrate signals
        if return_trajectories:
            X_history = [X.clone()]          # Record feature trajectories

        # Clean implementation without dropout and additional residual blocks in the loop
        for i in range(self.nlayers):
            X, Y = self.symp_Euler(X, Y, x_dp_l_fixed, x_dp_r_fixed, edge_index, norm, omega, zeta, Ks, omega_square)
            # Stablize Dyn. Systems
            X = self.norm_X(X)
            Y = self.norm_Y(Y)

            if return_trajectories:
                 X_history.append(X.clone())  # Record feature trajectories

        # Output dropouts
        X = F.dropout(X, self.dropout, training=self.training)
        X = self.dec(X)

        if return_trajectories:
            return X, X_history
        else:
            return X

    def symp_Euler(self, X, Y, x_dp_l_fixed, x_dp_r_fixed, edge_index, norm, omega, zeta, Ks, omega_square):
        # calculate coupling terms of CDS via signed attentions
        coupling, _ = self.coupling_block(Y, edge_index, x_dp_l_fixed, x_dp_r_fixed, edge_weight=norm)
        coupling_Y = self.linear_out(coupling)

        # symplectic Euler update step
        Y = Y + self.dt * (-2 * zeta * omega * Y - omega_square * X + Ks * coupling_Y)
        X = X + self.dt * Y

        return X, Y



class GraphCON_GAT(nn.Module):
    """ Code adapted from 
    https://github.com/tk-rusch/GraphCON/blob/main/src/heterophilic_graphs/models.py"""

    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, e_dropout=0.0, dt=1., zeta=1.0, omega=1., Ks=1, nheads=4, learnable_dyn_params=False):
        super(GraphCON_GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.nhid = nhid

        # ODE Hyperparameters
        self.nlayers = nlayers
        self.dt = dt
        self.Ks_raw = nn.Parameter(torch.tensor(Ks), requires_grad=learnable_dyn_params)  
        self.omega_raw = nn.Parameter(torch.tensor(omega), requires_grad=learnable_dyn_params)
        self.zeta_raw = nn.Parameter(torch.tensor(zeta), requires_grad=learnable_dyn_params)

        self.act_fn = nn.ReLU()
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GATConv(nhid, nhid, heads=nheads, dropout=e_dropout)
        self.dec = nn.Linear(nhid,nclass)


        self.norm_X = Norm(nhid, eps=1e-5)
        self.norm_Y = Norm(nhid, eps=1e-5)


    def forward(self, x, edge_index, return_trajectories=False):
        input = x
        n_nodes = x.size(0)
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        omega = F.softplus(self.omega_raw)   
        zeta  = F.softplus(self.zeta_raw)
        Ks    = F.softplus(self.Ks_raw)
        omega_square = omega ** 2 

        if return_trajectories:
            X_history = [X.clone()]          # ← 新增：记录初始特征

        for i in range(self.nlayers):
            Y = Y + self.dt*(Ks * F.elu(self.conv(X, edge_index)).view(n_nodes, -1, self.nheads).mean(dim=-1) - 2*zeta*omega*Y - omega_square*X)
            X = X + self.dt*Y
            X = self.norm_X(X)
            Y = self.norm_Y(Y)
            if return_trajectories:
                 X_history.append(X.clone())  # ← 每一步都记录

        X = F.dropout(X, self.dropout, training=self.training)
        X = self.dec(X)

        if return_trajectories:
            return X, X_history
        else:
            return X

class Kuramoto_GAT(nn.Module):
    """ Code adapted from 
    https://github.com/Fsoft-AIC/Reducing-Over-smoothing-via-a-Kuramoto-Model-based-Approach"""
    def __init__(self, nfeat, nhid, nclass, dropout, e_dropout, nlayers, nheads=8, dt=1, Ks=1, learnable_dyn_params=False):
        super(Kuramoto_GAT, self).__init__()
        # ODE Hyperparameters
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.Ks_raw = nn.Parameter(torch.tensor(Ks), requires_grad=learnable_dyn_params)  
        self.dt=dt

        # Networks
        self.enc = nn.Linear(nfeat,nhid)
        self.dec = nn.Linear(nhid,nclass)

        self.act_fn = nn.ReLU()
        self.att_block = GATatt(in_channels=nhid, out_channels=nhid, heads=nheads, add_self_loops=False, dropout=e_dropout)

        self.reset_params()
        

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)


    def sparse_multiply(self, x, mean_attention, edge_index):

            N = x.shape[0]
            E = edge_index.shape[1]
            

            indices = edge_index
            sparse_A = torch.sparse_coo_tensor(
                indices, 
                mean_attention, 
                size=(N, N), 
                device=x.device
            )

            phi   = torch.sparse.mm(sparse_A, x)
            cos_R = torch.sparse.mm(sparse_A, torch.cos(x))
            sin_R = torch.sparse.mm(sparse_A, torch.sin(x))

            return phi, cos_R, sin_R


    def forward(self, x, edge_index):
        input = x
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        omega = torch.clamp(Y.clone(), min=0, max=torch.pi)
        Y = F.dropout(Y, self.dropout, training=self.training)
        
        alpha = self.att_block(Y, edge_index)
        mean_attention = alpha.mean(dim=1)  

        Ks = F.softplus(self.Ks_raw)

        for _ in range(self.nlayers):

            # Calculate the coupling terms
            phi, cos_R, sin_R = self.sparse_multiply(Y,mean_attention,edge_index)
            R = torch.sqrt(cos_R**2 + sin_R**2)
            out_phi = phi-Y
            out_hat = Ks*R*torch.sin(out_phi)
            f = omega + out_hat
            Y = Y + self.dt*f

        out = Y

        out = F.dropout(out, self.dropout, training=self.training)
        out = self.dec(out)

        return out