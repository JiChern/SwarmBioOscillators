import copy
import sys,os,csv
import numpy as np
import torch
import torch.nn.functional as F

# from .utils import ReplayPool
from .replay_buffer import ReplayPoolGraph
from .networks import Policy, DoubleQFunc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3_Agent:

    def __init__(self, seed, state_dim, action_dim, action_lim=1, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256, 
                 update_interval=2, buffer_size=1e6, target_noise=0.2, target_noise_clip=0.5, explore_noise=0.1, optimize_alpha=True, heads=1, feature_dim=512):

        """Initialize the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
        
        This method sets up the core components of the TD3 agent, including actor/critic networks, 
        optimizers, replay buffer, and hyperparameters for training. It also configures exploration 
        noise and target network update settings.

        Args:
            seed (int): Random seed for PyTorch operations (e.g., weight initialization, noise generation). 
                Ensures reproducibility of training results.
            state_dim (int): Dimensionality of the environment's state space (e.g., 8 for an 8-dimensional observation).
            action_dim (int): Dimensionality of the environment's action space (e.g., 2 for a 2D robotic joint).
            action_lim (float, optional): Maximum absolute value for actions output by the actor. 
                Actions are scaled to [-action_lim, action_lim]. Defaults to 1.
            lr (float, optional): Learning rate for actor and critic Adam optimizers. Defaults to 3e-4.
            gamma (float, optional): Discount factor for future rewards (0-1). Balances immediate vs long-term rewards. 
                Defaults to 0.99.
            tau (float, optional): Soft update coefficient for target networks (0-1). 
                Controls how much target weights are updated per step (e.g., 0.005 = 0.5% of difference). Defaults to 5e-3.
            batchsize (int, optional): Number of samples drawn from the replay buffer per training iteration. 
                Larger batches improve gradient estimation. Defaults to 256.
            hidden_size (int, optional): Number of neurons in each hidden layer of [critic] networks. 
                Defines network capacity for learning complex mappings. Defaults to 256.
            update_interval (int, optional): Frequency (in steps) for updating the actor network. 
                TD3 uses delayed updates (e.g., every 2 critic updates) to stabilize training. Defaults to 2.
            buffer_size (int, optional): Capacity of the replay buffer (number of state-action-reward-next_state tuples). 
                Larger buffers improve sample efficiency. Defaults to 1e6.
            target_noise (float, optional): Standard deviation of Gaussian noise added to target actions during critic updates. 
                Smooths Q-value estimates to reduce overestimation. Defaults to 0.2.
            target_noise_clip (float, optional): Clips target noise to [-target_noise_clip, target_noise_clip]. 
                Prevents extreme noise from destabilizing training. Defaults to 0.5.
            explore_noise (float, optional): Standard deviation of Gaussian noise added to actions during execution. 
                Encourages exploration of the action space. Defaults to 0.1.
            optimize_alpha (bool, optional): If True, optimize the parameters for attention weights' generation ,
                If False, alpha is fixed. Defaults to True.
            heads (int, optional): Number of attention heads in the actor. 
                Defaults to 1.
            feature_dim (int, optional): Dimensionality of the shared feature representation extracted by the actor's encoder. 
                Higher dimensions capture more complex state patterns. Defaults to 512.
        """

        self.gamma = gamma
        self.tau = tau
        self.batchsize = batchsize
        self.update_interval = update_interval
        self.action_lim = action_lim

        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.explore_noise = explore_noise

        torch.manual_seed(seed)

        # aka critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, heads=heads,feature_dim=feature_dim).to(device)
        self.target_policy = copy.deepcopy(self.policy)

        self.policy.network.optimize_alpha = optimize_alpha
        self.target_policy.network.optimize_alpha = optimize_alpha

        for p in self.target_policy.parameters():
            p.requires_grad = False

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)  # lr=lr
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Use the customized replay buffer for advanced mini-batching for learning the graph neural network
        self.replay_pool = ReplayPoolGraph(capacity=int(buffer_size))

        self._update_counter = 0


    # def reallocate_replay_pool(self, new_size: int):
    #     assert new_size != self.replay_pool.capacity, "Error, you've tried to allocate a new pool which has the same length"
    #     new_replay_pool = ReplayPool(capacity=new_size)
    #     new_replay_pool.initialise(self.replay_pool)
    #     self.replay_pool = new_replay_pool

    def get_action(self, state, edge_index, deterministic=False, alpha_noise=None):
        """Get the action for the current state using the RL agent's policy network.

        This function processes the input state through the agent's policy network to generate an action. 
        It supports both stochastic (exploration) and deterministic (exploitation) modes, applies action 
        noise for exploration, and clamps the action to the predefined limits. The function also handles 
        device management (e.g., moving tensors to GPU/CPU) and ensures the output is in a usable format.

        Args:
            state (np.ndarray or torch.Tensor): The current state of the environment. Can be a NumPy array 
                or PyTorch tensor; will be converted to a PyTorch tensor and moved to the target device.  
            edge_index (torch.Tensor): The edge index representing graph-structured relationships in the state 
                (used for graph-based policies).  
            deterministic (bool, optional): If True, returns the deterministic action (mean of the policy distribution). 
                If False, adds exploration noise to the action. Defaults to False.  
            alpha_noise (float, optional): Scaling factor for exploration noise of attention weights. Defaults to None.  

        Returns:
            np.ndarray: The generated action, clamped to the range [-action_lim, action_lim] and converted to a 
                NumPy array with at least one dimension (e.g., [0.5] or [[-0.1, 0.3]] for multi-dimensional actions).  

        """

        state = torch.Tensor(state).to(device)

        if not deterministic:
            # Noise that directly added to attention weights for exploration
            with torch.no_grad():
                action = self.policy(state, edge_index, alpha_noise)
        else:
            with torch.no_grad():
                action = self.policy(state, edge_index, alpha_noise=None)

        # Noise that directly added to action for exploration
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)

        action.clamp_(-self.action_lim, self.action_lim)

        return np.atleast_1d(action.squeeze().cpu().numpy())
    
    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            for target_pi_param, pi_param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_pi_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * target_pi_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch, next_graph_x_batch, edge_index_batch):
        """Update the Q-functions (critic networks) using the TD3 algorithm's twin-delayed approach.
        
        This function computes the target Q-values using the Bellman equation with double Q-learning (to mitigate
        overestimation bias) and target policy smoothing (to add noise for exploration stability). It then updates
        the two Q-networks by minimizing the mean squared error (MSE) between their predictions and the target Q-values.
        The target networks are updated using soft updates (exponential moving average) to ensure stable training.

        Args:
            state_batch (torch.Tensor): Batch of current states from the replay buffer. Shape: (batch_size, state_dim).
            action_batch (torch.Tensor): Batch of actions taken in the current states. Shape: (batch_size, action_dim).
            reward_batch (torch.Tensor): Batch of immediate rewards received after taking `action_batch` in `state_batch`.
                Shape: (batch_size,).
            nextstate_batch (torch.Tensor): Batch of next states resulting from executing `action_batch` in `state_batch`.
                Shape: (batch_size, state_dim).
            done_batch (torch.Tensor): Batch of termination flags (1 if the episode ended after the step, 0 otherwise).
                Shape: (batch_size,).
            next_graph_x_batch (torch.Tensor): Batch of graph-structured features for next states (e.g., node embeddings).
                Used by graph-based policies to capture relational dependencies. Shape: (batch_size*num_nodes, graph_feature_dim).
            edge_index_batch (torch.Tensor): Batch of edge indices representing graph connectivity for next states.
                Shape: (2, num_edges*batch_size) (PyTorch Geometric format).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: MSE losses for the two Q-networks (q_1_loss, q_2_loss) computed against the target Q-values.
        """
                
        with torch.no_grad():
            nextaction_batch = self.target_policy(next_graph_x_batch,edge_index_batch).view(self.batchsize,-1)
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target

        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)

        return loss_1, loss_2

    def update_policy(self, state_batch, graph_x_batch, edge_index_batch):

        """Update the actor network (policy) using the TD3 algorithm's deterministic policy gradient approach.
        
        This function computes the policy loss by evaluating the current policy's actions with the twin Q-networks 
        (to mitigate overestimation bias) and updates the actor parameters to maximize the expected Q-value. The TD3-specific
        modification of using the minimum Q-value from two critics ensures more stable policy updates. The loss is computed as 
        the negative mean of the minimum Q-values, which the actor seeks to maximize via gradient ascent.

        The actor network (policy) takes graph-structured inputs (graph_x_batch, edge_index_batch) to generate actions, 
        which are then passed through the Q-networks to evaluate their quality. This approach is critical for continuous 
        control tasks where actions are high-dimensional and require precise optimization.

        Args:
            state_batch (torch.Tensor): Batch of current states from the replay buffer. Shape: (batch_size, state_dim).
            graph_x_batch (torch.Tensor): Batch of graph-structured features for the current states (e.g., node embeddings).
                Used by graph-based policies to capture relational dependencies. Shape: (batch_size*num_nodes, graph_feature_dim).
            edge_index_batch (torch.Tensor): Batch of edge indices representing graph connectivity for the current states.
                Shape: (2, batch_size*num_edges) (PyTorch Geometric format).

        Returns:
            torch.Tensor: The computed policy loss (scalar), which is the negative mean of the minimum Q-values from the two 
            Q-networks. This loss is used to update the actor network via backpropagation.
        """

        action_batch = self.policy(graph_x_batch, edge_index_batch).view(self.batchsize,-1)

        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (-qval_batch).mean()
        return policy_loss

    def optimize(self, n_updates):

        """Perform multiple optimization steps for the TD3 agent's actor and critic networks.
        
        This function executes `n_updates` iterations of optimization, where each iteration involves:
        1. Sampling a batch of experiences from the replay buffer.
        2. Updating the critic networks (Q-functions) to minimize the Bellman error between predicted and target Q-values.
        3. Periodically updating the actor network (policy) to maximize the Q-value of the current policy's actions.
        4. Applying delayed updates to the target networks (actor and critics) using Polyak averaging for stability.


        Args:
            n_updates (int): Number of optimization steps (iterations) to perform.

        Returns:
            tuple[float, float, float]: Cumulative losses for the two critic networks (q1_loss, q2_loss) and the actor network (pi_loss) 
                over all `n_updates` steps. These losses represent the average error between predicted and target values during optimization.
        """

        q1_loss, q2_loss, pi_loss = 0, 0, None

        for i in range(n_updates):

            samples = self.replay_pool.sample(self.batchsize)

            state_batch = torch.from_numpy(np.asarray(samples.state, dtype=np.float32)).to(device)
            nextstate_batch = torch.from_numpy(np.asarray(samples.next_state, dtype=np.float32)).to(device)
            graph_x_batch = torch.FloatTensor(samples.x).to(device)
            next_graph_x_batch = torch.FloatTensor(samples.next_x).to(device)
            edge_index_batch = samples.edge_index.to(device)


            action_batch = torch.from_numpy(np.asarray(samples.action, dtype=np.float32)).to(device).view(self.batchsize,-1)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.real_done.float()).to(device).unsqueeze(1)

            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch=state_batch,
                                                                 action_batch=action_batch,
                                                                 reward_batch=reward_batch,
                                                                 nextstate_batch=nextstate_batch,
                                                                 done_batch=done_batch,
                                                                 next_graph_x_batch=next_graph_x_batch,
                                                                 edge_index_batch=edge_index_batch)
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()
            
            self._update_counter += 1

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()

            if self._update_counter % self.update_interval == 0:
                if not pi_loss:
                    pi_loss = 0
                # update policy
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step = self.update_policy(state_batch, graph_x_batch, edge_index_batch)
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()

                self.policy_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True
                # update target policy and q-functions using Polyak averaging
                self.update_target()
                pi_loss += pi_loss_step.detach().item()


        return q1_loss, q2_loss, pi_loss
