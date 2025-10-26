import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool,GatedGraphConv

# from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric

from  torch_geometric.nn.dense.linear import Linear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from .actor_net import ActorNet



class MLPNetwork(nn.Module):
    """A standard Multi-Layer Perceptron (MLP) network for use in TD3.
    
    This network is composed of fully connected (Linear) layers with ReLU activations. It is used 
    as the base architecture for both Q-function networks (DoubleQFunc) and can be extended for other 
    purposes (e.g., value networks, policy embeddings). The network maps inputs to outputs through 
    a series of hidden layers with non-linear activations.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        hidden_size (int, optional): Number of neurons in each hidden layer. Defaults to 256.
    """    

    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()

        # Define a sequential network with 3 hidden layers and ReLU activations
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        """Forward pass: compute the output of the MLP given an input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
    

class DoubleQFunc(nn.Module):
    """A twin Q-function network for TD3, consisting of two identical MLP networks.
    
    TD3 uses two Q-networks (Q1 and Q2) to mitigate overestimation bias in Q-value estimation. 
    Both networks share the same architecture (MLPNetwork) but maintain separate parameters. 
    They take concatenated state and action inputs and output independent Q-value estimates.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_size (int, optional): Number of neurons in hidden layers of the MLP. Defaults to 256.
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        # Q1 network: maps (state || action) to a single Q-value
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        # Q2 network: maps (state || action) to a single Q-value
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        """Forward pass: compute Q-values from state and action using both Q-networks.
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Q-value tensors from Q1 and Q2 networks, each of shape (batch_size, 1).
        """

        # Concatenate state and action along the feature dimension
        x = torch.cat((state, action), dim=1)

        # Compute Q-values from both networks
        return self.network1(x), self.network2(x)

class Policy(nn.Module):
    """A Graph Neural Network (GNN) based policy network (actor) for TD3 in graph-structured environments.
    
    This policy network uses a custom GNN (ActorNet) to process graph-structured inputs (node features and edge indices) 
    and outputs continuous actions. It is designed for environments where the state is represented as a graph (e.g., 
    coupled oscillators, molecular structures). The network applies message passing, aggregates node features, and 
    produces actions through a linear output layer with tanh activation to ensure bounded actions.

    Args:
        state_dim (int): Dimension of the node feature space (e.g., 2 for x,y coordinates).
        action_dim (int): Dimension of the action space (e.g., 2 for controlling x,y directions).
        heads (int, optional): Number of attention heads in the GNN (if applicable). Defaults to 1.
        feature_dim (int): Dimension of the intermediate feature representation learned by the GNN. Defaults to 512.
    """

    def __init__(self, state_dim, action_dim, heads=1, feature_dim=512):
        super(Policy, self).__init__()
        self.action_dim = action_dim

        # Custom GNN-based message passing network for graph-CPG (coupled oscillator) environments
        # Input: node features (x_state: 2D), auxiliary features (x_dp: 2D), edge_index
        self.network = ActorNet(in_channels=4,   # Total input features per node (e.g., 4 = 2 state + 2 desired phase encoding)
                                state_channels=2,   # State features per node (e.g., 2D coordinates)
                                out_channels=feature_dim,  # Output feature dimension (intermediate representation)
                                heads=heads,      # Number of attention heads
                                add_self_loops=False,   # Do not add self-loops to edges
                                residual=False,    # No residual connections
                                dropout=0.2)  # Dropout for regularization
        
        # Linear output layer to map GNN features to action space
        # Input: feature_dim, Output: action_dim (e.g., 2)
        self.linear_out = Linear(in_channels=feature_dim, out_channels=2,bias=False, weight_initializer='glorot')


    def forward(self, x, edge_index, alpha_noise=None):
        """Forward pass: compute actions from graph-structured inputs using the GNN policy.
        
        Args:
            x (torch.Tensor): Node feature tensor of shape (num_nodes, 4), where columns 0-1 are state (x,y) 
                              and columns 2-3 are auxiliary features (e.g., velocities or phases).
            edge_index (torch.Tensor): Edge index tensor defining graph connectivity (PyTorch Geometric format).
            alpha_noise (torch.Tensor, optional): Noise for attention coefficients (if applicable). Defaults to None.
        
        Returns:
            torch.Tensor: Output action tensor of shape (num_nodes * action_dim,) or (batch_size * action_dim,).
                          Actions are squashed using tanh and flattened.
        """

        # Split input features into state (x_state: 0-1) and auxiliary (x_dp: 2-3)
        x_dp = x[:,2:]  # Auxiliary features (e.g., velocities)
        x_state = x[:,0:2]  # State features (e.g., x,y coordinates)

        # Process graph through the GNN (ActorNet) to obtain node embeddings
        x_coupling = self.network(x=x_state, x_dp=x_dp, edge_index=edge_index, alpha_noise=alpha_noise)
        
        # Squeeze unnecessary dimensions and map to action space
        x_coupling = torch.squeeze(x_coupling)
        x_coupling = self.linear_out(x_coupling)
        x_coupling = torch.squeeze(x_coupling)

        out = x_coupling
        
        # Apply tanh activation to bound actions (e.g., to [-1, 1])
        out = F.tanh(out)  
        
        # Flatten the output (e.g., for batch processing)
        out = out.ravel()
        return out

