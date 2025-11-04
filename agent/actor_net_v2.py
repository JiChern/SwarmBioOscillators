import typing


'''
Optional[X] is equivalent to Union[X, None]; Tuple[T1, T2] is a tuple of two elements
corresponding  to type variables T1 and T2; Union[X, Y] means either X or Y.
'''
from typing import Optional, Tuple, Union  

from torch_geometric.nn.inits import glorot, zeros  

import torch  
import torch.nn as nn  
from torch_geometric.nn import MessagePassing  
from torch_geometric.nn.dense.linear import Linear  
import torch.nn.functional as F  

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


# Type checking block for overload definitions (helps with IDEs and type checkers)
if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload  # For method overloading in TorchScript


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetV2(MessagePassing):
    """
    An SCPG implementation for ablation studies, extending PyTorch Geometric's MessagePassing.
    
    This class implements a graph neural network (GNN) actor that processes graph-structured state inputs (node features 
    and edge attributes) to output continuous actions. It combines desired phase lags with positional encodings (from 
    node-level features) and uses attention mechanisms (GATv2-style) for adaptive message passing. The network 
    supports residual connections, layer normalization, and configurable attention heads for multi-modal action learning.

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
        super().__init__(node_dim=0, **kwargs)  # Initialize base MessagePassing class with node_dim=0 (no broadcasting over nodes)

        # Store input/output and architecture configurations
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

        self.alpha_value = None  # Placeholder for storing attention values (for debugging or analysis)

        # Define linear projections for state features (single head output)
        if isinstance(state_channels, int):
            self.lin_l = Linear(state_channels, 1 * out_channels, bias=bias,
                    weight_initializer='glorot')  # Left (source) projection
            if share_weights:
                self.lin_r = self.lin_l  # Share weights for right (target) projection if enabled
            else:
                self.lin_r = Linear(state_channels, 1 * out_channels,
                                    bias=bias, weight_initializer='glorot')  # Separate right projection
        else:
            raise TypeError('in channels must be of int type')  # Ensure state_channels is integer
        
        

        
        # Compute total output channels based on concatenation mode
        total_out_channels = out_channels * (heads if concat else 1)


        # Define residual projection if enabled
        if residual:
            self.res = Linear(
                state_channels
                if isinstance(state_channels, int) else state_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)  # No residual if disabled

        # Define learnable bias if enabled
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Learnable attention parameter (for GATv2-style attention)
        # dp is considered to cope with attention parameters to produce egde weights
        self.att = Parameter(torch.empty(1, heads, out_channels))   


        # Projections for desired phase lags (multi-head output)
        self.lin_dp = Linear(in_channels-state_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')  # Left phase lag projection
        self.lin_dp_r = Linear(in_channels-state_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')  # Right phase lag projection
        self.x_state_layer = Linear(state_channels, 1 * out_channels, bias=bias, weight_initializer='glorot')  # Additional state projection layer    


        self.alpha_scaling = 1  # Scaling factor for attention coefficients
        

        self.negative_slope = 0.2  # Negative slope for LeakyReLU (overwritten from init arg)
        self.num_nodes = None  # Number of nodes (dynamically set later)
        

        self.reset_parameters()  # Initialize parameters

        self.optimize_alpha = True  # Flag to enable gradient computation for attention


    def reset_parameters(self):
        """Reset all learnable parameters using Glorot/Xavier initialization and zeros for bias."""
        super().reset_parameters()  # Call base reset
        self.lin_l.reset_parameters()  # Reset left projection
        self.lin_r.reset_parameters()  # Reset right projection
        
        if self.res is not None:
            self.res.reset_parameters()  # Reset residual if enabled
        glorot(self.att)  # Glorot init for attention parameter
        zeros(self.bias)  # Zero init for bias



    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        x_dp: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        alpha_noise = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass  # Overload for forward without attention weights return

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass  # Overload for forward with attention weights (dense edge_index)

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass  # Overload for forward with attention weights (sparse edge_index)


    def forward(  # noqa: F811
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
        """
        Forward pass of the actor network.
        
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
        
        Notes:
            - Assumes single-head for state projections and multi-head for phase lags (potential dimension mismatch if heads > 1).
            - Does not handle return_attention_weights (always returns output tensor).
        """


        H, C = self.heads, self.out_channels  # Heads and channels for multi-head attention
        Hx = 1  # Single-head for state projections

        res: Optional[Tensor] = None  # Placeholder for residual

        x_l: OptTensor = None  # Source node projections
        x_r: OptTensor = None  # Target node projections
        x_dp_l: OptTensor = None  # Source phase lag projections
        x_dp_r:  OptTensor = None  # Target phase lag projections


        self.state_vec = x  # Store input state vector (for debugging or access)
 
        # Project desired phase lags to multi-head hidden space
        x_dp_l = self.lin_dp(x_dp).view(-1, H, C)
        x_dp_r = self.lin_dp_r(x_dp).view(-1, H, C)
 
        # Project state to single-head hidden space
        x_state = self.x_state_layer(x).view(-1, H, C)  # Note: H=heads, but projection is 1 * out_channels; potential mismatch if H > 1

        # Add positional encoding (state) to phase lag projections
        x_dp_l = x_dp_l + x_state
        x_dp_r = x_dp_r + x_state


        # Process node features if input is a single tensor
        if isinstance(x, Tensor):
            assert x.dim() == 2  # Ensure 2D tensor (num_nodes, features)

            if self.res is not None:  # Compute residual if enabled
                res = self.res(x).view(-1, Hx, C)

            # Project node features to single-head hidden space
            x_l = self.lin_l(x).view(-1, Hx, C)
            if self.share_weights:
                x_r = x_l  # Share projections if enabled
            else:
                x_r = self.lin_r(x).view(-1, Hx, C)


        # Assertions to ensure projections are not None
        assert x_l is not None
        assert x_r is not None


        # Add self-loops if enabled
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)  # Remove existing self-loops
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)  # Add new self-loops
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)  # Set diagonal for sparse self-loops
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # Compute attention coefficients using edge updater
        if self.optimize_alpha:
            alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr)  # With gradients
        else:
            with torch.no_grad():
                alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr)  # Without gradients



        alpha = alpha * self.alpha_scaling  # Scale attention
            
        if alpha_noise is not None:
            alpha = alpha + alpha_noise  # Add noise if provided

        # Store first 12 attention values (transposed) for analysis (hardcoded slice; assumes specific graph size)
        self.alpha_value = torch.transpose(alpha.detach()[0:12],0,1)


        # Propagate messages using computed attention (source and target both use x_l)
        out = self.propagate(edge_index, x=(x_l, x_l), alpha=alpha) 


        # Add residual if enabled
        if res is not None:
            out = out + res

        # Add bias if enabled
        if self.bias is not None:
            out = out + self.bias

        
        return out  

         
    
    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        """
        Compute attention coefficients for edges.
        
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
            Tensor: Attention coefficients (alpha) of shape (num_edges, heads).
        """
        x = x_i + x_j  # Combine source and target features

        # Add edge attributes if provided (note: self.lin_edge is referenced but not defined in __init__; potential bug)
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None  # Assertion may fail if lin_edge not initialized
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr


        # Apply LeakyReLU activation
        x = F.leaky_relu(x, self.negative_slope)


        # Compute raw attention scores (matrix multiplication with att)
        alpha = (x * self.att)


        # Sum over channel dimension
        alpha = alpha.sum(dim=-1)


        # Apply tanh to squash scores to [-1, 1]
        alpha = F.tanh(alpha)

        # Apply dropout for regularization
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        """
        Message passing function.
        
        Weighted aggregation of neighbor features (x_j) using attention coefficients (alpha).

        Args:
            x_j (Tensor): Neighbor node features. Shape (num_edges, heads, out_channels).
            alpha (Tensor): Attention coefficients. Shape (num_edges, heads).
        
        Returns:
            Tensor: Weighted messages. Shape (num_edges, heads, out_channels).
        """

        return x_j * alpha.unsqueeze(-1)  # Broadcast alpha to multiply features

    def __repr__(self) -> str:
        """String representation of the actor network."""
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')