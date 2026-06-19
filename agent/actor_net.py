import typing
import sys, os, csv

'''
Optional[X] is equivalent to Union[X, None]; Tuple[T1, T2] is a tuple of two elements
corresponding  to type variables T1 and T2; Union[X, Y] means either X or Y.
'''

from typing import Optional, Tuple, Union  

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
    
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)

from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload

pi = np.pi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class SIES_Coupling(MessagePassing):

    """Implementation of SIES model, extending PyTorch Geometric's MessagePassing.
    
    This class implements a graph neural network (GNN) actor that processes graph-structured state inputs (node features 
    and edge attributes) to output continuous actions. It combines desired phase lags  with positional encodings (from 
    node-level features) and uses attention mechanisms (GATv2-style) for adaptive message passing. The network 
    supports residual connections, layer normalization, and configurable attention heads for multi-modal action learning.

    Key Features:
    - Graph Structured Inputs: Processes node features (`x`) and edge attributes (`edge_attr`) to model relational dependencies.
    - Attention Mechanisms: Uses learnable attention coefficients (via `edge_updater`) based on desired phase lags(x_dp) to weight message importance.
    - Positional Encodings: Incorporates node-level features to store the temporal information in attention weights.
    - Residual Connections: Optional skip connections to preserve original node features.
    - Multi-Head Attention: Supports multiple attention heads for capturing diverse information.

    Args:
        in_channels (Union[int, Tuple[int, int]]): Input feature dimension(s).
        state_channels (Union[int, Tuple[int, int]]): State feature dimension(s) for positional encoding (e.g., dynamic differences).
        out_channels (int): Output action dimension.
        heads (int, optional): Number of attention heads. Defaults to 1.
        concat (bool, optional): If True, concatenates outputs from all heads; else averages. Defaults to False.
        negative_slope (float, optional): Negative slope for LeakyReLU. Defaults to 0.2.
        dropout (float, optional): Dropout probability for attention coefficients. Defaults to 0.0.
        add_self_loops (bool, optional): If True, adds self-loops to edge indices. Defaults to False.
        edge_dim (Optional[int], optional): Edge feature dimension. Defaults to None.
        fill_value (Union[float, Tensor, str], optional): Value for self-loop edge attributes. Defaults to 'mean'.
        bias (bool, optional): If True, adds learnable bias to output. Defaults to False.
        share_weights (bool, optional): If True, shares weights between source/target node projections. Defaults to True.
        residual (bool, optional): If True, adds skip connection from input to output. Defaults to False.
        **kwargs: Additional arguments for PyTorch Geometric's MessagePassing.
    """

    def __init__(self,        
        in_channels: Union[int, Tuple[int, int]],
        state_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = False,
        share_weights: bool = True,
        residual: bool = False,
        signed_att = True,
        direction_aware = True,
        state_space_aggr = False,
        **kwargs,):
        super().__init__(node_dim=0, **kwargs)

        # Input/output and architecture configuration
        self.in_channels = in_channels
        self.state_channels = state_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        self.signed_att = signed_att
        self.direction_aware = direction_aware
        self.state_space_aggr = state_space_aggr
        self.alpha_value = None

        # Learnable attention parameter (for GATv2-style attention)
        self.att = Parameter(torch.empty(1, heads, out_channels))  

        # Linear projections for node features and positional encodings
        if isinstance(state_channels, int):
            self.lin_l = Linear(state_channels, 1* heads * out_channels, bias=bias,
                    weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(state_channels, 1 * heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            raise TypeError('in channels must be of int type')
        
        
        
        total_out_channels = out_channels * (heads if concat else 1)


        # Residual connection (if enabled)
        if residual:
            self.res = Linear(
                state_channels
                if isinstance(state_channels, int) else state_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        # Learnable bias (if enabled)
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)


        # Project desired phase lags to hidden dimension
        if self.direction_aware:
            self.lin_dp = Linear(in_channels-state_channels, 1 * heads * out_channels , bias=bias,
                                    weight_initializer='glorot')
            self.lin_dp_r = Linear(in_channels-state_channels, 1 * heads * out_channels , bias=bias,
                                    weight_initializer='glorot')
        else:
            self.lin_dp = Linear(in_channels-state_channels, 1 * heads * out_channels , bias=bias,
                                    weight_initializer='glorot')
            

        
        self.negative_slope = 0.2

        # Node count (fixed for learning environment)
        self.num_nodes = 8
        
        # Reset parameters (initializes weights)
        self.reset_parameters()


    def reset_parameters(self):
        """Reset all learnable parameters using Xavier/Glorot initialization."""
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        

        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)


    def forward(  
        self,
        x: Union[Tensor, PairTensor],
        x_dp: Union[Tensor, PairTensor],
        edge_index: Adj,   # Adj = Union[Tensor, SparseTensor]
        edge_attr: OptTensor = None,
        alpha_noise = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:

        """Forward pass of the actor network.
        
        Processes graph-structured inputs (node features `x`, desired phase lags `x_dp`, edge attributes `edge_attr`) 
        through message passing with attention, followed by aggregation and residual connections.

        Args:
            x (Union[Tensor, PairTensor]): Node features. If tensor, shape is (num_nodes, in_channels); if PairTensor, 
                contains separate source/target features for bipartite graphs.
            x_dp (Union[Tensor, PairTensor]): Desired phase lags. Shape matches `x`.
            edge_index (Adj): Graph connectivity (edge indices). Can be tensor (2, num_edges) or sparse tensor.
            edge_attr (OptTensor, optional): Edge features. Defaults to None.
            alpha_noise (Optional[Tensor], optional): Noise for attention coefficients. Defaults to None.
            return_attention_weights (Optional[bool], optional): If True, returns attention weights. Defaults to None.
        
        Returns:
            Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, SparseTensor]]: 
                - If `return_attention_weights=False`: Output node features of shape (num_nodes, out_channels) or 
                  (num_nodes, heads * out_channels) if `concat=True`.
                - If `return_attention_weights=True`: Tuple of (output features, attention weights).
        """


        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        x_dp_l: OptTensor = None
        x_dp_r:  OptTensor = None

        if self.direction_aware:
            x_dp_l = self.lin_dp(x_dp).view(-1, H, C)
            x_dp_r = self.lin_dp_r(x_dp).view(-1, H, C)
        else:
            x_dp_l = self.lin_dp(x_dp).view(-1, H, C)
            x_dp_r = self.lin_dp(x_dp).view(-1, H, C)

        if isinstance(x, Tensor):
            assert x.dim() == 2  # Input node features must be 2D tensor 

            # Apply residual connection if enabled
            if self.res is not None:  # skip connection if self.res is not None
                res = self.res(x).view(-1, H, C)

            # Project node features to hidden dimension
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        

        assert x_l is not None # Source node features cannot be None
        assert x_r is not None # Target node features cannot be None
        
        x_dp_l = x_dp_l + x_l
        x_dp_r = x_dp_r + x_r

        # Compute attention coefficients (alpha) via edge updater based on (desired phase lags + node features)
        if self.training:
            alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr, alpha_noise=alpha_noise)
        else:
            with torch.no_grad():
                alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr, alpha_noise=None)

        self.alpha_value = torch.transpose(alpha.detach(),0,1)


        row, col = edge_index  

        # deg norm similar to GCN
        if self.signed_att:
            deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [num_edges]
        else:
            edge_weight = None


        # Message aggregation
        if not self.state_space_aggr:
            x_j = x_r[row]  # [E, H, C]
            msg = self.message(x_j=x_j, alpha=alpha, edge_weight=edge_weight)  # 传入edge_weight
            num_nodes = x.size(0)
            out = torch.zeros((num_nodes, H, C), device=x.device, dtype=x.dtype)
            msg = msg.to(out.dtype)
            out = out.index_add(dim=0, index=col, source=msg)
        
        else:
            x_expand = x.unsqueeze(1)
            x_aggr = x_expand.expand(-1,H,-1)
            x_j = x_aggr[row]  # [E, H, C]
            msg = self.message(x_j=x_j, alpha=alpha, edge_weight=edge_weight)  # 传入edge_weight
            num_nodes = x.size(0)
            out = torch.zeros((num_nodes, H, 2), device=x.device, dtype=x.dtype)
            msg = msg.to(out.dtype)
            out = out.index_add(dim=0, index=col, source=msg)
            


        # Propagate messages (aggregate neighbor features) using attention
        # out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        # Post-process output: concatenate or average heads, add residual, and bias
        if self.concat:
            out = out.view(-1,self.heads*self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        
        return out
        
        
    
    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int], alpha_noise=None) -> Tensor:
        """Compute attention coefficients for edges.
        
        Aggregates node features (x_i, x_j) and edge attributes (if provided) to compute 
        unnormalized attention scores.

        Args:
            x_j (Tensor): Source node features. Shape (num_edges, heads, out_channels).
            x_i (Tensor): Target node features. Shape (num_edges, heads, out_channels).
            edge_attr (OptTensor, optional): Edge features. Defaults to None.
            index (Tensor): Edge indices. Shape (num_edges,).
            ptr (OptTensor, optional): Pointer for sparse edge indices. Defaults to None.
            dim_size (Optional[int], optional): Size of output tensor. Defaults to None.
        
        Returns:
            Tensor: Unnormalized attention scores (alpha) of shape (num_edges, heads).
        """


        # Combine source and target node features
        x = x_i + x_j

        # Add edge attributes if provided (in this study, edge_attr is set to None)
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        # Apply leaky ReLU activation
        x = F.leaky_relu(x, self.negative_slope)

        # Compute attention scores (dot product with learned attention vector)
        alpha = (x * self.att).sum(dim=-1)
        
        if alpha_noise is not None:
            alpha = alpha + alpha_noise
        
        # # Apply tanh to squash scores to [-1, 1]
        if not self.signed_att:
            alpha = softmax(alpha, index, ptr, dim_size)
        

        # Apply dropout for regularization
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        return alpha
    
    def message(self, x_j: Tensor, alpha: Tensor, edge_weight: OptTensor = None) -> Tensor:
        msg = x_j * alpha.unsqueeze(-1)
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1).unsqueeze(-1) 

        return msg


    def __repr__(self) -> str:
        """String representation of the actor network."""
        
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class SIES_Coupling_Softmax(MessagePassing):

    """Implementation of SIES model, extending PyTorch Geometric's MessagePassing.
    
    This class implements a graph neural network (GNN) actor that processes graph-structured state inputs (node features 
    and edge attributes) to output continuous actions. It combines desired phase lags  with positional encodings (from 
    node-level features) and uses attention mechanisms (GATv2-style) for adaptive message passing. The network 
    supports residual connections, layer normalization, and configurable attention heads for multi-modal action learning.

    Key Features:
    - Graph Structured Inputs: Processes node features (`x`) and edge attributes (`edge_attr`) to model relational dependencies.
    - Attention Mechanisms: Uses learnable attention coefficients (via `edge_updater`) based on desired phase lags(x_dp) to weight message importance.
    - Positional Encodings: Incorporates node-level features to store the temporal information in attention weights.
    - Residual Connections: Optional skip connections to preserve original node features.
    - Multi-Head Attention: Supports multiple attention heads for capturing diverse information.

    Args:
        in_channels (Union[int, Tuple[int, int]]): Input feature dimension(s).
        state_channels (Union[int, Tuple[int, int]]): State feature dimension(s) for positional encoding (e.g., dynamic differences).
        out_channels (int): Output action dimension.
        heads (int, optional): Number of attention heads. Defaults to 1.
        concat (bool, optional): If True, concatenates outputs from all heads; else averages. Defaults to False.
        negative_slope (float, optional): Negative slope for LeakyReLU. Defaults to 0.2.
        dropout (float, optional): Dropout probability for attention coefficients. Defaults to 0.0.
        add_self_loops (bool, optional): If True, adds self-loops to edge indices. Defaults to False.
        edge_dim (Optional[int], optional): Edge feature dimension. Defaults to None.
        fill_value (Union[float, Tensor, str], optional): Value for self-loop edge attributes. Defaults to 'mean'.
        bias (bool, optional): If True, adds learnable bias to output. Defaults to False.
        share_weights (bool, optional): If True, shares weights between source/target node projections. Defaults to True.
        residual (bool, optional): If True, adds skip connection from input to output. Defaults to False.
        **kwargs: Additional arguments for PyTorch Geometric's MessagePassing.
    """

    def __init__(self,        
        in_channels: Union[int, Tuple[int, int]],
        state_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = False,
        share_weights: bool = True,
        residual: bool = False,
        **kwargs,):
        super().__init__(node_dim=0, **kwargs)

        # Input/output and architecture configuration
        self.in_channels = in_channels
        self.state_channels = state_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        self.alpha_value = None

        # Learnable attention parameter (for GATv2-style attention)
        self.att = Parameter(torch.empty(1, heads, out_channels))  

        # Linear projections for node features and positional encodings
        if isinstance(state_channels, int):
            self.lin_l = Linear(state_channels, 1* heads * out_channels, bias=bias,
                    weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(state_channels, 1 * heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            raise TypeError('in channels must be of int type')
        

        
        total_out_channels = out_channels * (heads if concat else 1)


        # Residual connection (if enabled)
        if residual:
            self.res = Linear(
                state_channels
                if isinstance(state_channels, int) else state_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        # Learnable bias (if enabled)
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

         

        # Project desired phase lags to hidden dimension

        self.lin_dp = Linear(in_channels-state_channels, 1 * heads * out_channels , bias=bias,
                                weight_initializer='glorot')
        self.lin_dp_r = Linear(in_channels-state_channels, 1 * heads * out_channels , bias=bias,
                                weight_initializer='glorot')

        
        self.negative_slope = 0.2

        # Node count (fixed for learning environment)
        self.num_nodes = 8
        
        # Reset parameters (initializes weights)
        self.reset_parameters()

        # Optimization flag for attention coefficients
        self.optimize_alpha = True


    def reset_parameters(self):
        """Reset all learnable parameters using Xavier/Glorot initialization."""
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        

        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)


    def forward(  
        self,
        x: Union[Tensor, PairTensor],
        x_dp: Union[Tensor, PairTensor],
        edge_index: Adj,   # Adj = Union[Tensor, SparseTensor]
        edge_attr: OptTensor = None,
        alpha_noise = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:

        """Forward pass of the actor network.
        
        Processes graph-structured inputs (node features `x`, desired phase lags `x_dp`, edge attributes `edge_attr`) 
        through message passing with attention, followed by aggregation and residual connections.

        Args:
            x (Union[Tensor, PairTensor]): Node features. If tensor, shape is (num_nodes, in_channels); if PairTensor, 
                contains separate source/target features for bipartite graphs.
            x_dp (Union[Tensor, PairTensor]): Desired phase lags. Shape matches `x`.
            edge_index (Adj): Graph connectivity (edge indices). Can be tensor (2, num_edges) or sparse tensor.
            edge_attr (OptTensor, optional): Edge features. Defaults to None.
            alpha_noise (Optional[Tensor], optional): Noise for attention coefficients. Defaults to None.
            return_attention_weights (Optional[bool], optional): If True, returns attention weights. Defaults to None.
        
        Returns:
            Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, SparseTensor]]: 
                - If `return_attention_weights=False`: Output node features of shape (num_nodes, out_channels) or 
                  (num_nodes, heads * out_channels) if `concat=True`.
                - If `return_attention_weights=True`: Tuple of (output features, attention weights).
        """


        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        x_dp_l: OptTensor = None
        x_dp_r:  OptTensor = None

        x_dp_l = self.lin_dp(x_dp).view(-1, H, C)
        x_dp_r = self.lin_dp_r(x_dp).view(-1, H, C)


        if isinstance(x, Tensor):
            assert x.dim() == 2  # Input node features must be 2D tensor 

            # Apply residual connection if enabled
            if self.res is not None:  # skip connection if self.res is not None
                res = self.res(x).view(-1, H, C)

            # Project node features to hidden dimension
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        

        assert x_l is not None # Source node features cannot be None
        assert x_r is not None # Target node features cannot be None
        
        x_dp_l = x_dp_l + x_l
        x_dp_r = x_dp_r + x_r

        # Compute attention coefficients (alpha) via edge updater based on (desired phase lags + node features)
        if self.optimize_alpha:
            alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr, alpha_noise=alpha_noise)
        else:
            with torch.no_grad():
                alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr)

        

        self.alpha_value = torch.transpose(alpha.detach(),0,1)


        row, col = edge_index  # 注意: edge_index是Tensor时是[2, num_edges]

        edge_weight = None  # [num_edges]



        # Message aggregation
        x_j = x_r[row]  # [E, H, C]
        msg = self.message(x_j=x_j, alpha=alpha, edge_weight=edge_weight)  # 传入edge_weight
        num_nodes = x.size(0)
        out = torch.zeros((num_nodes, H, C), device=x.device, dtype=x.dtype)
        msg = msg.to(out.dtype)
        out = out.index_add(dim=0, index=col, source=msg)

        # Propagate messages (aggregate neighbor features) using attention
        # out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        # Post-process output: concatenate or average heads, add residual, and bias
        if self.concat:
            out = out.view(-1,self.heads*self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        
        return out
        
        
    
    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int], alpha_noise=None) -> Tensor:
        """Compute attention coefficients for edges.
        
        Aggregates node features (x_i, x_j) and edge attributes (if provided) to compute 
        unnormalized attention scores.

        Args:
            x_j (Tensor): Source node features. Shape (num_edges, heads, out_channels).
            x_i (Tensor): Target node features. Shape (num_edges, heads, out_channels).
            edge_attr (OptTensor, optional): Edge features. Defaults to None.
            index (Tensor): Edge indices. Shape (num_edges,).
            ptr (OptTensor, optional): Pointer for sparse edge indices. Defaults to None.
            dim_size (Optional[int], optional): Size of output tensor. Defaults to None.
        
        Returns:
            Tensor: Unnormalized attention scores (alpha) of shape (num_edges, heads).
        """


        # Combine source and target node features
        x = x_i + x_j

        # Add edge attributes if provided (in this study, edge_attr is set to None)
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        # Apply leaky ReLU activation
        x = F.leaky_relu(x, self.negative_slope)

        # Compute attention scores (dot product with learned attention vector)
        alpha = (x * self.att).sum(dim=-1)

        if alpha_noise is not None:
            alpha = alpha + alpha_noise
        
        # # Apply tanh to squash scores to [-1, 1]
        alpha = softmax(alpha, index, ptr, dim_size)

        # Apply dropout for regularization
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        return alpha
    
    def message(self, x_j: Tensor, alpha: Tensor, edge_weight: OptTensor = None) -> Tensor:
        msg = x_j * alpha.unsqueeze(-1)
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1).unsqueeze(-1) 

        return msg


    def __repr__(self) -> str:
        """String representation of the actor network."""
        
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


