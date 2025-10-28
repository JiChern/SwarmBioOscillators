import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool,GatedGraphConv

# from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric

from  torch_geometric.nn.dense.linear import Linear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from .actor_net import ActorNet
from .actor_net_state_space import ActorNetStateSpace
from .actor_net_v2 import ActorNetV2



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
    
    This policy network uses a custom GNN (ActorNet,graph-CPG) to process graph-structured inputs (node features and edge indices) 
    and outputs continuous actions. It is designed for environments where the state is represented as a graph (e.g., 
    coupled oscillators, molecular structures). The network applies message passing, aggregates node features, and 
    produces actions through a linear output layer with tanh activation to ensure bounded actions.

    Args:
        heads (int, optional): Number of attention heads in the GNN (if applicable). Defaults to 1.
        feature_dim (int): Dimension of the intermediate feature representation learned by the GNN. Defaults to 512.
    """

    def __init__(self, heads=1, feature_dim=512):
        super(Policy, self).__init__()
        # Custom GNN-based message passing network for graph-CPG environments
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


class PolicyV2(nn.Module):
    """ 
    Standard graph-CPG model of ablation studies, aggregate features in state space.

    Notes:
    - Input `x` shape: (num_nodes, 4), where first 2 columns are state features (e.g., x, y oscillator states),
      and last 2 are desired phase lags (e.g., sin, cos encodings).
    - Output: Flattened tensor of shape (num_nodes * 2), representing actions (e.g., coupling adjustments).
    - Used in RL or control pipelines for multi-agent/oscillator synchronization.
    """

    def __init__(self):
        """
        Initialize the PolicyV2 network.

        Sets up the internal GNN actor and output linear layer with specific hyperparameters.
        - ActorNetV2: Configured for 4 input channels (2 state + 2 phase), 512 hidden channels, single head.
        - No residual connections, dropout=0.2 for regularization.
        - Output linear layer: Maps 512 hidden features to 2D action space without bias.
        """
        super(PolicyV2, self).__init__()  # Call base nn.Module initializer
        # Initialize the core GNN actor network with specified parameters
        self.network = ActorNetV2(in_channels=4, state_channels=2, out_channels=512, heads=1, add_self_loops=False, residual=False, dropout=0.2)  # without dropout
        # Define output linear projection from hidden dimension to action space (2D)
        self.linear_out = Linear(in_channels=512, out_channels=2, bias=False, weight_initializer='glorot')


    def forward(self, x, edge_index, alpha_noise=None):
        """
        Forward pass of the policy network.

        Args:
            x (torch.Tensor): Input node features. Shape (num_nodes, 4), where:
                - Columns 0-1: State features (x_state, e.g., oscillator coordinates).
                - Columns 2-3: Desired phase lag encodings (x_dp, e.g., sin/cos).
            edge_index (torch.Tensor or SparseTensor): Graph edge indices defining connectivity.
            alpha_noise (Optional[torch.Tensor], optional): Noise to add to attention coefficients (for exploration). Defaults to None.

        Returns:
            torch.Tensor: Flattened action tensor. Shape (num_nodes * 2), bounded to [-1, 1] via tanh.

        Notes:
            - Splits input into state and phase components for ActorNetV2.
            - Applies tanh activation for action bounding.
            - Output is raveled (flattened) for compatibility with environments expecting 1D actions.
        """

        # Split input features into phase lags (x_dp) and states (x_state)
        x_dp = x[:,2:]  # Desired phase lags (columns 2 and beyond)
        x_state = x[:,0:2]  # State features (first two columns)


        # Compute coupling terms using the GNN actor (message passing with attention)
        x_coupling = self.network(x=x_state, x_dp=x_dp, edge_index=edge_index, alpha_noise=alpha_noise)
        # Remove any singleton dimensions (e.g., from batch or head squeezing)
        x_coupling = torch.squeeze(x_coupling)
        # Project coupling features to action space
        x_coupling = self.linear_out(x_coupling)
        # Squeeze again to ensure correct shape (num_nodes, 2)
        x_coupling = torch.squeeze(x_coupling)

        # Assign to output (for clarity; could be combined)
        out = x_coupling

        # Apply tanh activation to bound outputs to [-1, 1] (common for continuous actions)
        out = F.tanh(out)
        # Flatten the output tensor to 1D (e.g., for environment compatibility)
        out = out.ravel()
        return out

class PolicyStateSpace(nn.Module):
    """ 
        V0 model of ablation studies, aggregate features in state space.
    Args:
        heads (int, optional): Number of attention heads in the GNN (if applicable). Defaults to 1.
        feature_dim (int): Dimension of the intermediate feature representation learned by the GNN. Defaults to 512.
    """

    def __init__(self, heads=1, feature_dim=512):
        super(PolicyStateSpace, self).__init__()
        # Custom GNN-based message passing network for graph-CPG (coupled oscillator) environments
        # Input: node features (x_state: 2D), auxiliary features (x_dp: 2D), edge_index
        self.network = ActorNetStateSpace(in_channels=4,   # Total input features per node (e.g., 4 = 2 state + 2 desired phase encoding)
                                state_channels=2,   # State features per node (e.g., 2D coordinates)
                                out_channels=feature_dim,  # Output feature dimension (intermediate representation)
                                heads=heads,      # Number of attention heads
                                add_self_loops=False,   # Do not add self-loops to edges
                                residual=False,    # No residual connections
                                dropout=0.2)  # Dropout for regularization
        

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

        # Simplified graph-CPG directly outputs the action
        x_coupling = self.network(x=x_state, x_dp=x_dp, edge_index=edge_index, alpha_noise=alpha_noise)

        out = x_coupling
        
        # Apply tanh activation to bound actions (e.g., to [-1, 1])
        out = F.tanh(out)  
        
        # Flatten the output (e.g., for batch processing)
        out = out.ravel()
        return out

class PolicyMLP(nn.Module):
    """Forward pass: V1 model of ablation studies, computes actions from MLP policy.
    
    Args:
        x (torch.Tensor): Node feature tensor of shape (num_nodes x 4)
    
    Returns:
        torch.Tensor: Output action tensor of shape (num_nodes * action_dim,) or (batch_size * action_dim,).
                        Actions are squashed using tanh and flattened.
    """ 

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(PolicyMLP, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.network(x)
        x = self.tanh(x)

        return x