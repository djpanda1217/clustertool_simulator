from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
# import yaml

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn

from datetime import datetime, timedelta
import argparse
import itertools

import os
import EQP_Scheduler_env_AB
import seaborn as sns
from scipy import stats

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.pos = 0

    def append(self, transition):
        max_prio = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(max_prio)
        else:
            self.memory[self.pos] = transition
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            raise ValueError("PER buffer is empty!")

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        return samples, indices, weights

    def update_priorities(self, indices, td_errors, epsilon=1e-5):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error.item()) + epsilon

    def __len__(self):
        return len(self.memory)

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs_AB_DuelingDQN"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Q-Learning Agent
class Agent():

    def __init__(self):
       
        # Hyperparameters (adjustable)
        self.env_id             = 'eqp-scheduler-v0_AB_3_FP'
        self.learning_rate_a    = 0.0005                                    # learning rate (alpha) 지금수정
        self.discount_factor_g  = 0.99                                      # discount rate (gamma)
        self.network_sync_rate  = 10000                                     # number of steps the agent takes before syncing the policy and target network 지금수정
        self.replay_memory_size = 10000                                     # size of replay memory
        self.mini_batch_size    = 32                                        # size of the training data set sampled from the replay memory
        self.epsilon_init       = 1                                         # 1 = 100% random actions
        self.epsilon_decay      = 0.9995                                    # epsilon decay rate
        self.epsilon_min        = 0.05                                      # minimum epsilon value
        self.stop_on_reward     = 100000                                    # stop training after reaching this number of rewards
        self.fc1_nodes          = 1024

        # Neural Network
        self.loss_fn = nn.SmoothL1Loss()                                    # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                                               # NN Optimizer. Initialize later.

        # Get current date and time
        current_time = datetime.now().strftime('%Y%m%d-%H%M')

        # Update env_id with date and time
        self.file_name = f'{current_time}-{self.env_id}'

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.file_name}.log')
        self.MODEL_FILE_VTR = os.path.join(RUNS_DIR, f'{self.file_name}_vtr.pt')
        self.MODEL_FILE_ATR = os.path.join(RUNS_DIR, f'{self.file_name}_atr.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.file_name}.png')

    def run(self, trained_model_file='', is_training=True, render=False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None)

        if isinstance(env.action_space, gym.spaces.Tuple):
            num_vtr_actions = env.action_space.spaces[0].n
            num_atr_actions = env.action_space.spaces[1].n
            print(f"VTR actions: {num_vtr_actions}, ATR actions: {num_atr_actions}")
        else:
            raise ValueError("Unexpected action space structure")

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        vtr_policy_dqn = DQN(num_states, num_vtr_actions, self.fc1_nodes).to(device)
        atr_policy_dqn = DQN(num_states, num_atr_actions, self.fc1_nodes).to(device)

        if is_training:
            epsilon = self.epsilon_init
            self.memory = PrioritizedReplayMemory(self.replay_memory_size)

            vtr_target_dqn = DQN(num_states, num_vtr_actions, self.fc1_nodes).to(device)
            atr_target_dqn = DQN(num_states, num_atr_actions, self.fc1_nodes).to(device)
            vtr_target_dqn.load_state_dict(vtr_policy_dqn.state_dict())
            atr_target_dqn.load_state_dict(atr_policy_dqn.state_dict())

            self.vtr_optimizer = torch.optim.Adam(vtr_policy_dqn.parameters(), lr=self.learning_rate_a)
            self.atr_optimizer = torch.optim.Adam(atr_policy_dqn.parameters(), lr=self.learning_rate_a)

            self.vtr_scheduler = torch.optim.lr_scheduler.StepLR(self.vtr_optimizer, step_size=5000, gamma=0.5)
            self.atr_scheduler = torch.optim.lr_scheduler.StepLR(self.atr_optimizer, step_size=5000, gamma=0.5)

            epsilon_history = []
            step_count = 0
            best_reward = -9999999
        else:
            vtr_policy_dqn.load_state_dict(torch.load(os.path.join(RUNS_DIR, f'{trained_model_file}_vtr.pt')))
            atr_policy_dqn.load_state_dict(torch.load(os.path.join(RUNS_DIR, f'{trained_model_file}_atr.pt')))
            vtr_policy_dqn.eval()
            atr_policy_dqn.eval()

        avg_wait_times_per_episode = []

        for episode in range(10000):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                invalid_vtr_actions, invalid_atr_actions = env.unwrapped.invalid_actions()
                
                if is_training and random.random() < epsilon:
                    vtr_action = random.choice([a for a in range(num_vtr_actions) if a not in invalid_vtr_actions])
                    atr_action = random.choice([a for a in range(num_atr_actions) if a not in invalid_atr_actions])
                else:
                    with torch.no_grad():
                        vtr_q_values = vtr_policy_dqn(state.unsqueeze(dim=0)).squeeze()
                        atr_q_values = atr_policy_dqn(state.unsqueeze(dim=0)).squeeze()
                        vtr_q_values[invalid_vtr_actions] = float('-inf')
                        atr_q_values[invalid_atr_actions] = float('-inf')
                        vtr_action = vtr_q_values.argmax().item()
                        atr_action = atr_q_values.argmax().item()

                action = (vtr_action, atr_action)
                new_state, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    self.memory.append((state, vtr_action, atr_action, new_state, reward, terminated))
                    step_count += 1

                state = new_state


            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)
            avg_wait_times_per_episode.append(info['avg_processed_wait_time'])
            info_formatted = {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in info.items()}
            if is_training:
                print(f"Episode {episode}, Reward: {episode_reward:.3f}, info: {info_formatted}, Epsilon: {epsilon:.3f}")
            else:
                print(f"Episode {episode}, Reward: {episode_reward:.3f}, info: {info_formatted}")


            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                    vtr_model_file = os.path.join(RUNS_DIR, f'{self.file_name}_ep{episode}_{timestamp}_vtr.pt')
                    atr_model_file = os.path.join(RUNS_DIR, f'{self.file_name}_ep{episode}_{timestamp}_atr.pt')

                    torch.save(vtr_policy_dqn.state_dict(), vtr_model_file)
                    torch.save(atr_policy_dqn.state_dict(), atr_model_file)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=3):
                    self.save_graph(rewards_per_episode, epsilon_history, avg_wait_times_per_episode)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(self.memory) > self.mini_batch_size:
                    # mini_batch = memory.sample(self.mini_batch_size)
                    mini_batch_data = self.memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch_data, vtr_policy_dqn, atr_policy_dqn, vtr_target_dqn, atr_target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        vtr_target_dqn.load_state_dict(vtr_policy_dqn.state_dict())
                        atr_target_dqn.load_state_dict(atr_policy_dqn.state_dict())
                        step_count = 0


    def save_graph(self, rewards_per_episode, epsilon_history, avg_wait_times_per_episode):
        # Save plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        ax1.plot(mean_rewards)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Mean Rewards')
        ax1.set_title('Average Rewards')

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        ax2.plot(epsilon_history)
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Epsilon Decay')

        # Plot average wait time
        ax3.plot(avg_wait_times_per_episode)
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Avg Wait Time')
        ax3.set_title('Average Wait Time')

        plt.tight_layout()

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Optimize policy network
    def optimize(self, mini_batch_data, vtr_policy_dqn, atr_policy_dqn, vtr_target_dqn, atr_target_dqn):
        mini_batch, indices, weights = mini_batch_data
        states, vtr_actions, atr_actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        vtr_actions = torch.tensor(vtr_actions, dtype=torch.long, device=device)
        atr_actions = torch.tensor(atr_actions, dtype=torch.long, device=device)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            vtr_next_actions = vtr_policy_dqn(new_states).argmax(dim=1, keepdim=True)
            atr_next_actions = atr_policy_dqn(new_states).argmax(dim=1, keepdim=True)

            vtr_target_q_values = vtr_target_dqn(new_states)
            atr_target_q_values = atr_target_dqn(new_states)

            vtr_target_q = rewards + (1 - terminations) * self.discount_factor_g * vtr_target_q_values.gather(1, vtr_next_actions).squeeze(1)
            atr_target_q = rewards + (1 - terminations) * self.discount_factor_g * atr_target_q_values.gather(1, atr_next_actions).squeeze(1)

        vtr_current_q = vtr_policy_dqn(states).gather(1, vtr_actions.unsqueeze(1)).squeeze(1)
        atr_current_q = atr_policy_dqn(states).gather(1, atr_actions.unsqueeze(1)).squeeze(1)

        vtr_td_error = (vtr_current_q - vtr_target_q).detach().abs()
        atr_td_error = (atr_current_q - atr_target_q).detach().abs()
        avg_td_error = (vtr_td_error + atr_td_error) / 2.0
        self.memory.update_priorities(indices, avg_td_error)

        vtr_loss = torch.mean(weights * F.smooth_l1_loss(vtr_current_q, vtr_target_q, reduction='none'))
        atr_loss = torch.mean(weights * F.smooth_l1_loss(atr_current_q, atr_target_q, reduction='none'))

        self.vtr_optimizer.zero_grad()
        self.atr_optimizer.zero_grad()
        vtr_loss.backward()
        atr_loss.backward()

        torch.nn.utils.clip_grad_norm_(vtr_policy_dqn.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(atr_policy_dqn.parameters(), max_norm=1.0)

        self.vtr_optimizer.step()
        self.atr_optimizer.step()

        self.vtr_scheduler.step()
        self.atr_scheduler.step()

    def evaluate(self, num_episodes=10, render=False):
        env = gym.make(self.env_id, render_mode='human' if render else None)
        if isinstance(env.action_space, gym.spaces.Tuple):
            num_vtr_actions = env.action_space.spaces[0].n
            num_atr_actions = env.action_space.spaces[1].n
            print(f"VTR actions: {num_vtr_actions}, ATR actions: {num_atr_actions}")
        else:
            raise ValueError("Unexpected action space structure")


        model1_atr_path = '/workspace/DQN_E3_FP/runs_AB_1/20250218-0419-eqp-scheduler-v0_AB_3_FP_ep72564_20250218-123115_atr.pt'
        model1_vtr_path = '/workspace/DQN_E3_FP/runs_AB_1/20250218-0419-eqp-scheduler-v0_AB_3_FP_ep72564_20250218-123115_vtr.pt'

        policy_dqn_model1_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model1_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model1_atr.load_state_dict(torch.load(model1_atr_path))
        policy_dqn_model1_vtr.load_state_dict(torch.load(model1_vtr_path))
        policy_dqn_model1_atr.eval()
        policy_dqn_model1_vtr.eval()

        dummy_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        dummy_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)

        model1_wafers = []
        random_wafers = []
        optimal1_wafers = []
        optimal2_wafers = []

        for _ in range(num_episodes):
            model1_wafer = self.run_episode(env, policy_dqn_model1_atr, policy_dqn_model1_vtr, use_model=True)
            random_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 0)
            optimal1_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 1)
            optimal2_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 2)
            
            model1_wafers.append(model1_wafer)
            random_wafers.append(random_wafer)
            optimal1_wafers.append(optimal1_wafer)
            optimal2_wafers.append(optimal2_wafer)

        env.close()
        return model1_wafers, random_wafers, optimal1_wafers, optimal2_wafers

    def run_episode(self, env, policy_dqn_atr, policy_dqn_vtr, use_model=True, rule_based_type = 0):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        
        done = False

        while not done:
            invalid_vtr_actions, invalid_atr_actions = env.unwrapped.invalid_actions()
            
            if use_model:
                with torch.no_grad():
                    vtr_q_values = policy_dqn_vtr(state.unsqueeze(dim=0)).squeeze()
                    atr_q_values = policy_dqn_atr(state.unsqueeze(dim=0)).squeeze()
                    vtr_q_values[invalid_vtr_actions] = float('-inf')
                    atr_q_values[invalid_atr_actions] = float('-inf')
                    vtr_action = vtr_q_values.argmax().item()
                    atr_action = atr_q_values.argmax().item()
            else:
                # Random
                if rule_based_type == 0:
                    vtr_action = random.choice([a for a in range(env.action_space.spaces[0].n) if a not in invalid_vtr_actions])
                    atr_action = random.choice([a for a in range(env.action_space.spaces[1].n) if a not in invalid_atr_actions])
                # Rule 1
                elif rule_based_type == 1:
                    
                    vtr_action_space_size = env.action_space.spaces[0].n
                    if set(range(1, vtr_action_space_size)) - set(invalid_vtr_actions):
                        invalid_vtr_actions = list(set(invalid_vtr_actions) | {0})

                    atr_action_space_size = env.action_space.spaces[1].n
                    if set(range(1, atr_action_space_size)) - set(invalid_atr_actions):
                        invalid_atr_actions = list(set(invalid_atr_actions) | {0})


                    vtr_action = random.choice([a for a in range(env.action_space.spaces[0].n) if a not in invalid_vtr_actions])
                    atr_action = random.choice([a for a in range(env.action_space.spaces[1].n) if a not in invalid_atr_actions])

                # Rule 2
                elif rule_based_type == 2:
                    vtr_action_space_size = env.action_space.spaces[0].n
                    if set(range(1, vtr_action_space_size)) - set(invalid_vtr_actions):
                        invalid_vtr_actions = list(set(invalid_vtr_actions) | {0})

                    atr_action_space_size = env.action_space.spaces[1].n
                    if set(range(1, atr_action_space_size)) - set(invalid_atr_actions):
                        invalid_atr_actions = list(set(invalid_atr_actions) | {0})

                    valid_vtr_actions = [a for a in range(vtr_action_space_size) if a not in invalid_vtr_actions]
                    valid_atr_actions = [a for a in range(atr_action_space_size) if a not in invalid_atr_actions]
                    
                    vtr_action = env.unwrapped.get_min_exe_time_vtr_action(valid_vtr_actions)
                    atr_action = env.unwrapped.get_min_exe_time_atr_action(valid_atr_actions)
                

            action = (vtr_action, atr_action)
            state, _, done, _, info = env.step(action)
            state = torch.tensor(state, dtype=torch.float, device=device)

        return info['total_wafer_processed']

    def compare_and_plot(self, num_episodes=10):
        model_wafers, random_wafers, optimal1_wafers, optimal2_wafers = self.evaluate(num_episodes)

        print(f"---------- Result ----------")
        print(f"Model | Mean: {np.mean(model_wafers):.2f}, Std: {np.std(model_wafers):.2f}")
        print(f"Random Sampling | Mean: {np.mean(random_wafers):.2f}, Std: {np.std(random_wafers):.2f}")
        print(f"Optimal 1 | Mean: {np.mean(optimal1_wafers):.2f}, Std: {np.std(optimal1_wafers):.2f}")
        print(f"Optimal 2 | Mean: {np.mean(optimal2_wafers):.2f}, Std: {np.std(optimal2_wafers):.2f}")

        

if __name__ == '__main__':
    dql = Agent()
    dql.run(is_training=True)
    #dql.run(trained_model_file='20241021-1559-eqp-scheduler-v0', is_training=False, render=True)
    # dql.compare_and_plot(num_episodes=10)