import typing

'''
Optional[X] is equivalent to Union[X, None]; Tuple[T1, T2] is a tuple of two elements
corresponding  to type variables T1 and T2; Union[X, Y] means either X or Y.
'''
from typing import Optional, Tuple, Union  # Type hints for better code readability and static analysis

from torch_geometric.nn.inits import glorot, zeros  # Initialization functions for weights (Glorot/Xavier and zeros)

import torch  
import torch.nn as nn  
from torch_geometric.nn import MessagePassing  # Graph neural network layers (MessagePassing is base class)
from torch_geometric.nn.dense.linear import Linear  # Dense linear layer for projections
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


# Type checking for overloads (aids IDEs and type checkers)
if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload  # For method overloading in TorchScript



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetStateSpace(MessagePassing):
    """
    Actor network variant for Graph-CPG that aggregates node information directly in state space.
    
    This class is a specialized version of the actor network where message passing and aggregation 
    occur directly in the state space (e.g., oscillator coordinates) rather than a projected hidden space. 
    It uses attention mechanisms to weight neighbor contributions but propagates raw state features (x_l, x_r) 
    during message passing. This design focuses on state-space aggregation for tasks like phase synchronization 
    in CPGs, potentially simplifying the model for ablation studies or efficiency.

    Key Differences from Standard ActorNet:
    - States (x) are directly reshaped and propagated without full projection (used as-is in message passing).
    - Phase lags (x_dp) are projected and added as positional encodings to compute attention.
    - Hardcoded for 8 nodes (self.num_nodes=8).
    - Simplified edge_update (no separate leaky_x storage; direct computation).

    Args:
        in_channels (Union[int, Tuple[int, int]]): Input feature dimension(s).
        state_channels (Union[int, Tuple[int, int]]): State feature dimension(s).
        out_channels (int): Output action dimension.
        heads (int, optional): Number of attention heads. Defaults to 1.
        concat (bool, optional): If True, concatenates heads; else averages. Defaults to False.
        negative_slope (float, optional): LeakyReLU slope. Defaults to 0.2.
        dropout (float, optional): Dropout for attention. Defaults to 0.0.
        add_self_loops (bool, optional): Add self-loops. Defaults to False.
        edge_dim (Optional[int], optional): Edge feature dim. Defaults to None.
        fill_value (Union[float, Tensor, str], optional): Self-loop fill value. Defaults to 'mean'.
        bias (bool, optional): Add bias. Defaults to False.
        share_weights (bool, optional): Share source/target weights. Defaults to True.
        residual (bool, optional): Add residual connection. Defaults to False.
        **kwargs: Additional MessagePassing args.
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
        super().__init__(node_dim=0, **kwargs)  # Initialize base MessagePassing

        # Store configurations
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

        self.alpha_value = None  # Placeholder for attention values

        

        # State projections (single-head)
        if isinstance(state_channels, int):
            self.lin_l = Linear(state_channels, 1 * out_channels, bias=bias,
                    weight_initializer='glorot')  # Source projection
            if share_weights:
                self.lin_r = self.lin_l
            else:

                self.lin_r = Linear(state_channels, 1 * out_channels,
                                    bias=bias, weight_initializer='glorot')  # Target projection
        else:
            raise TypeError('in channels must be of int type')  # Validate input type
        
        

        
        # Total output channels (considering concat)
        total_out_channels = out_channels * (heads if concat else 1)


        # Residual projection if enabled
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

        # Bias parameter
        if bias:
            # self.bias = Parameter(torch.empty(total_out_channels))
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        
        # Attention parameter
        self.att = Parameter(torch.empty(1, heads, out_channels))   

        # Phase lag projections (multi-head)
        # xdp is considered to cope with attention parameters to produce egde weights
        self.lin_dp = Linear(in_channels-state_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
        self.lin_dp_r = Linear(in_channels-state_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
        self.x_state_layer = Linear(state_channels, 1 * out_channels, bias=bias, weight_initializer='glorot')    



        self.alpha_scaling = 1  # Attention scaling factor
        

        self.negative_slope = 0.2  # LeakyReLU slope (overwritten)
        self.num_nodes = 8  # Hardcoded number of nodes (assumes fixed graph size)
        

        self.reset_parameters()  # Init parameters

        self.variable_dict = {'x':None, 'leacky_x': None}  # Debug storage

        self.optimize_alpha = True  # Enable attention gradients


    def reset_parameters(self):
        """Reset parameters with Glorot and zeros."""
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)


    def cal_phase_relation(self,x):
        """Compute relative phase lags for 6 cells (hardcoded)."""
        phase_lag = torch.zeros(6)
        for i in range(6):
            phase_lag[i] = x[i] - x[0]
        return phase_lag


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
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass


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


        H, C = self.heads, self.out_channels
        Hx = 1

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        x_dp_l: OptTensor = None
        x_dp_r:  OptTensor = None


        # Project phase lags
        x_dp_l = self.lin_dp(x_dp).view(-1, H, C)
        x_dp_r = self.lin_dp_r(x_dp).view(-1, H, C)

        # Project state
        x_state = self.x_state_layer(x).view(-1, H, C)

        # Add positional encoding
        x_dp_l = x_dp_l + x_state
        x_dp_r = x_dp_r + x_state

                
        # Directly use states for propagation (aggregation in state space)
        x_l = x.view(-1, 1, 2)
        x_r = x.view(-1, 1, 2)
        

        assert x_l is not None
        assert x_r is not None


        # Add self-loops if enabled
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # Compute attention
        if self.optimize_alpha:
            alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr)
        else:
            with torch.no_grad():
                alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr)

        alpha = alpha * self.alpha_scaling
            
        if alpha_noise is not None:
            # print(alpha_noise)
            alpha = alpha + alpha_noise



        self.alpha_value = torch.transpose(alpha.detach()[0:12],0,1)

        # Propagate using raw states and attention
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)



        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        
        return out
        
        
    
    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        """Compute attention coefficients."""

        x = x_i + x_j

        # Edge attr handling (lin_edge undefined; potential bug)
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        self.variable_dict['x'] = x

        x = F.leaky_relu(x, self.negative_slope)

        # Compute alpha
        alpha = (x * self.att)

        alpha = alpha.sum(dim=-1)

        alpha = F.tanh(alpha)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        """Weighted messages."""
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')