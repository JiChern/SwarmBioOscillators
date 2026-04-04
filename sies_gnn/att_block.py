from typing import Optional, Union

import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
import math
from torch_geometric.utils import degree


class SIES_coupling_block(MessagePassing):
    """
    A separate class for computing signed attention coefficients based on node features and edge indices.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
        share_weights: bool = True,
        concat: bool = False,
        **kwargs,
        
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.share_weights = share_weights
        self.concat = concat

        # Learnable attention parameter (for GATv2-style attention)
        self.att = Parameter(torch.empty(1, heads, hidden_channels))  

        total_hidden_channels = hidden_channels * (heads if concat else 1)

        # Project desired phase lags to hidden dimension
        self.lin_dp = Linear(in_channels, 1 * heads * hidden_channels, bias=bias,
                             weight_initializer='glorot')
        self.lin_dp_r = Linear(in_channels, 1 * heads * hidden_channels, bias=bias,
                               weight_initializer='glorot')

        # Linear projections for node features
        self.lin_l = Linear(in_channels, heads * hidden_channels, bias=bias, weight_initializer='glorot')
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * hidden_channels, bias=bias, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters using Xavier/Glorot initialization."""
        glorot(self.att)
        self.lin_l.reset_parameters()
        self.lin_dp.reset_parameters()
        self.lin_dp_r.reset_parameters()
        if not self.share_weights:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        x_dp_l_fixed,
        x_dp_r_fixed,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,  
    ) -> Tensor:
        """
        Compute attention coefficients based on node features and edge indices.

        Args:
            x (Union[Tensor, PairTensor]): Node features. Shape (num_nodes, in_channels).
            edge_index (Adj): Graph connectivity (edge indices).
            edge_attr (OptTensor, optional): Edge features. Defaults to None.
            edge_weight (OptTensor, optional): Per-edge weights for normalization. If None, compute internally.

        Returns:
            Tensor: Output features, alpha (attention coefficients).
        """

        H, C = self.heads, self.hidden_channels
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = self.lin_r(x).view(-1, H, C)
        else:
            raise ValueError("Input x must be a single Tensor for node features.")

        # Combine projected desired phase lags with projected node features (positional encoding)   
        x_dp_l = x_dp_l_fixed + x_l
        x_dp_r = x_dp_r_fixed + x_r


        row, col = edge_index  # 注意: edge_index是Tensor时是[2, num_edges]

        # If edge_weight is not provided，calculate GCN-style normalization
        if edge_weight is None:
            deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [num_edges]

        # Compute attention coefficients using edge updater
        alpha = self.edge_updater(edge_index, x=(x_dp_l, x_dp_r), edge_attr=edge_attr)

        # Message aggregation
        x_j = x_r[row]  # [E, H, C]
        msg = self.message(x_j=x_j, alpha=alpha, edge_weight=edge_weight)  # 传入edge_weight
        num_nodes = x.size(0)
        out = torch.zeros((num_nodes, H, C), device=x.device, dtype=x.dtype)
        msg = msg.to(out.dtype)
        out = out.index_add(dim=0, index=col, source=msg)

        # Post-process output: concatenate or average heads
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        return out, alpha

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """Compute attention coefficients for edges."""

        # GATv2 style attention without softmax post processing (direct outputs signed attention coeffs)
        x = x_i + x_j
        x = F.leaky_relu(x, negative_slope=0.2)
        alpha = (x * self.att).sum(dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha
    
    def message(self, x_j: Tensor, alpha: Tensor, edge_weight: OptTensor = None) -> Tensor:
        msg = x_j * alpha.unsqueeze(-1)
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1).unsqueeze(-1) 


        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, heads={self.heads})')

