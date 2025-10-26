import sys, os, csv
import random
from argparse import ArgumentParser
from collections import deque
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show

import numpy as np
import torch


from agent.td3 import TD3_Agent
from utils import make_checkpoint,rearrange_state_vector_hopf
# from env import CPGEnv
from environment.env import CPGEnv


np. set_printoptions(precision=2)
np.set_printoptions(threshold=np.inf)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# coupled oscillator system with synaptic structure
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 3, 5, 7],
                           [1, 0, 3, 2, 5, 4, 7, 6, 2, 4, 6, 0, 3, 5, 7, 1]], dtype=torch.long).to(device='cuda')



from utils import state_to_goal


from torch_geometric.data import Data
from torch_geometric.typing import TensorFrame, torch_frame

data_folder = os.path.join(sys.path[0], 'TD3/data')



def train_agent_model_free(agent, env, env_eval, params):

    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = 1000     # Logging for every 1000 samples.
    save_interval = 50000   # Save the model checkpoints for every 50000 samples.
    n_collect_steps = params['n_collect_steps']  # The number samples that must be collected before the training
    save_model = params['save_model']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []


    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    noise_decay = 1
    alpha_noise_scalar = 10

    # Initialize progress bar and time tracking
    max_samples = int(2e6)
    start_time = time.time()
    last_time = start_time
    
    # Create progress bar
    pbar = tqdm(total=max_samples, desc="Training", unit="samples")
    pbar.set_postfix({"Episode": 0, "Reward": 0, "ETA": "calculating..."})

    while samples_number < max_samples:

        time_step = 0
        episode_reward = 0
        i_episode += 1
        log_episode += 1

        # Assign equal probility for choosing a goal in an episode
        env.prob = np.array([1/4,1/4,1/4,1/4])        
        env_eval.prob = np.array([1/4,1/4,1/4,1/4])   

        # reset the goal and initial state of one run
        env.reset_goal()          
        state = env.reset(ini_state=None)             


        done = False

        # Set the noise which is directly added to coupling weights
        noise_size = torch.zeros(16,1).to(device='cuda')  
        alpha_noise =  alpha_noise_scalar * torch.randn_like(noise_size)
        alpha_noise_scalar = alpha_noise_scalar * noise_decay

        while (not done):
            
            cumulative_log_timestep += 1
            cumulative_timestep += 1
            time_step += 1
            samples_number += 1

            
            assert not np.any(np.isnan(state)) 

            # Convert the state vector (with dim 4Nx1) to input matrix (with dim Nx4) of graph neural network 
            gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=8)

            # The action is the graph-CPG's influence to the coupled oscillator network
            action = agent.get_action(gnn_x, edge_index, alpha_noise=alpha_noise)

            # Propogate the one step in environment based on the action to get the nextstate and reward
            nextstate, reward, done, _= env.step(action)
            real_done = False
            
            # Convert the nextstate vector (with dim 4Nx1) to input matrix (with dim Nx4) of graph neural network 
            next_gnn_x = rearrange_state_vector_hopf(nextstate, num_nodes=8)

            # This the transition tuple of reinforcement learning.
            # Here we construct it with data structure of pytorch Geometry, becase training graph neural networks needs more information
            data = Data(x=gnn_x,
                        edge_index=edge_index, 
                        next_x = next_gnn_x, 
                        state=state, 
                        action=action, 
                        reward=reward, 
                        next_state=nextstate, 
                        real_done=real_done)
            
            # Save the transition tuple to the replay buffer
            agent.replay_pool.push(data)


            state = nextstate
            
            # Calculate the accumulated reward of an episode
            episode_reward += reward

            # Update network parameters if it's time
            if cumulative_timestep % 1000 == 0 and cumulative_timestep > n_collect_steps:    # step RL optimization per 1000 steps
                agent.optimize(update_timestep)
                n_updates += 1

            # Update progress bar
            if samples_number % 1000 == 0:  # Update every 1000 steps
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Calculate estimated remaining time
                if samples_number > 0:
                    time_per_sample = elapsed_time / samples_number
                    remaining_samples = max_samples - samples_number
                    eta_seconds = time_per_sample * remaining_samples
                    eta = str(timedelta(seconds=int(eta_seconds)))
                else:
                    eta = "calculating..."
                
                # Update progress bar
                pbar.update(samples_number - pbar.n)
                current_time_str = datetime.now().strftime("%H:%M:%S")
                pbar.set_postfix({
                    "Episode": i_episode,
                    "Reward": f"{episode_reward:.2f}",
                    "Time": current_time_str,
                    "ETA": eta
                })

            # Evaluation and logging
            if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps: # n_collect_steps = 1000
                avg_length = np.mean(episode_steps)
                running_reward = np.mean(episode_rewards)
                agent.policy.train()

                dp, reward_sum = evaluate_agent(env_eval, agent)

                # Beautified print format
                print("\n" + "="*80)
                print(f"📊 TRAINING METRICS - Episode {i_episode}")
                print("="*80)
                print(f"📈 Samples:        {samples_number:,}")
                print(f"🎯 Episode:        {i_episode}")
                print(f"📏 Avg Length:     {avg_length:.2f}")
                print(f"🏋️  Train Reward:   {running_reward:.4f}")
                print(f"🎯 Test Reward:    {reward_sum:.4f}")
                print(f"🔄 Updates:        {n_updates}")
                print(f"🧠 Heads:          {params['heads']}")
                print(f"🔧 Feature Dim:    {params['fd']}")
                print("="*80 + "\n")
                
                # Concise log format
                log_msg = f"Episode {i_episode} | Samples {samples_number:,} | Train {running_reward:.4f} | Test {reward_sum:.4f} | Updates {n_updates}"
                logging.info(log_msg)
                
                episode_steps = []
                episode_rewards = []
            
            # Save the model checkpoints for further evaluations.
            if cumulative_timestep % save_interval == 0:
                if save_model:
                    checkpoint_name = 'multi-goal-'+str(params['heads'])+'-'+str(params['fd'])
                    make_checkpoint(agent, cumulative_timestep, params['env'], checkpoint_name)
                    logging.info(f'Model checkpoint saved at step {cumulative_timestep}: {checkpoint_name}')
        
        episode_steps.append(time_step)
        episode_rewards.append(episode_reward)
    
    # Close progress bar
    pbar.close()
    total_time = timedelta(seconds=int(time.time() - start_time))
    completion_msg = f"Training completed! Total time: {total_time}"
    print(f"\n{completion_msg}")
    logging.info(completion_msg)
    logging.info(f"Final stats: Total samples: {samples_number}, Total episodes: {i_episode}, Total updates: {n_updates}")
        
        
        



def evaluate_agent(env, agent):
    cwd = os.getcwd()
    data_folder = cwd+'/data'

    reward_sum = 0
    done = False

    goal, goal_index = env.reset_goal()
    state = env.reset(ini_state=None)

    desired_pl = env.desired_lag
    while (not done):
        gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=8)
        action = agent.get_action(gnn_x, edge_index, deterministic=True)
        nextstate, reward, done, _ = env.step(action)

        reward_sum += reward
        state = nextstate

    return desired_pl, reward_sum


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='CPG_r_i')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=20)
    parser.add_argument('--n_random_actions', type=int, default=25000)
    parser.add_argument('--n_collect_steps', type=int, default=3000)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--fd', type=int, default=512)

    
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=True)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    heads = params['heads']
    fd = params['fd']

    # Setup logging
    log_filename = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}_seed{seed}_heads{heads}_fd{fd}.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also output to terminal
        ]
    )
    
    logging.info(f"Starting training with parameters: seed={seed}, heads={heads}, fd={fd}")
    logging.info(f"Log file: {log_filename}")

    # construct the RL environments of 8-cell-coupled oscillators, the maximum steps of and episode is 125.
    env = CPGEnv(cell_nums=8, env_length=125)  
    env_eval = CPGEnv(cell_nums=8, env_length=125)

    # construct the RL agent based on twin-delayed-DDPG.
    agent = TD3_Agent(seed, state_dim=16+16, action_dim=16, batchsize=512, explore_noise=0.01, lr=4e-5, gamma=0.995, heads=heads, feature_dim = fd) #lr=4e-5

    train_agent_model_free(agent=agent, env=env, env_eval=env_eval, params=params)


if __name__ == '__main__':
    main()
