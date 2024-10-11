from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from datetime import datetime, timedelta
import argparse
import itertools

import os
import EQP_Scheduler_env
import seaborn as sns
from scipy import stats

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

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return self.output(x)

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Q-Learning Agent
class Agent():

    def __init__(self):
       
        # Hyperparameters (adjustable)
        self.env_id             = 'eqp-scheduler-v0'
        self.learning_rate_a    = 0.0001                                    # learning rate (alpha)
        self.discount_factor_g  = 0.99                                      # discount rate (gamma)
        self.network_sync_rate  = 10                                        # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 100000                                    # size of replay memory
        self.mini_batch_size    = 32                                        # size of the training data set sampled from the replay memory
        self.epsilon_init       = 1                                         # 1 = 100% random actions
        self.epsilon_decay      = 0.99                                      # epsilon decay rate
        self.epsilon_min        = 0.05                                      # minimum epsilon value
        self.stop_on_reward     = 100000                                    # stop training after reaching this number of rewards
        self.fc1_nodes          = 512

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

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

        # 수정된 부분: action_space가 Tuple인 경우를 처리
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

        # VTR과 ATR에 대한 별도의 DQN 생성
        vtr_policy_dqn = DQN(num_states, num_vtr_actions, self.fc1_nodes).to(device)
        atr_policy_dqn = DQN(num_states, num_atr_actions, self.fc1_nodes).to(device)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)

            vtr_target_dqn = DQN(num_states, num_vtr_actions, self.fc1_nodes).to(device)
            atr_target_dqn = DQN(num_states, num_atr_actions, self.fc1_nodes).to(device)
            vtr_target_dqn.load_state_dict(vtr_policy_dqn.state_dict())
            atr_target_dqn.load_state_dict(atr_policy_dqn.state_dict())

            self.vtr_optimizer = torch.optim.Adam(vtr_policy_dqn.parameters(), lr=self.learning_rate_a)
            self.atr_optimizer = torch.optim.Adam(atr_policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon_history = []
            step_count = 0
            best_reward = -9999999
        else:
            vtr_policy_dqn.load_state_dict(torch.load(os.path.join(RUNS_DIR, f'{trained_model_file}_vtr.pt')))
            atr_policy_dqn.load_state_dict(torch.load(os.path.join(RUNS_DIR, f'{trained_model_file}_atr.pt')))
            vtr_policy_dqn.eval()
            atr_policy_dqn.eval()

        avg_wait_times_per_episode = []

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                invalid_vtr_actions, invalid_atr_actions = env.invalid_actions()
                
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
                    memory.append((state, vtr_action, atr_action, new_state, reward, terminated))
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

                    torch.save(vtr_policy_dqn.state_dict(), self.MODEL_FILE_VTR)
                    torch.save(atr_policy_dqn.state_dict(), self.MODEL_FILE_ATR)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=3):
                    self.save_graph(rewards_per_episode, epsilon_history, avg_wait_times_per_episode)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, vtr_policy_dqn, atr_policy_dqn, vtr_target_dqn, atr_target_dqn)

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

        # 평균 대기 시간 그래프
        ax3.plot(avg_wait_times_per_episode)
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Avg Wait Time')
        ax3.set_title('Average Wait Time')

        plt.tight_layout()

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Optimize policy network
    def optimize(self, mini_batch, vtr_policy_dqn, atr_policy_dqn, vtr_target_dqn, atr_target_dqn):
        states, vtr_actions, atr_actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        vtr_actions = torch.tensor(vtr_actions, dtype=torch.long, device=device)
        atr_actions = torch.tensor(atr_actions, dtype=torch.long, device=device)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            vtr_target_q = rewards + (1-terminations) * self.discount_factor_g * vtr_target_dqn(new_states).max(dim=1)[0]
            atr_target_q = rewards + (1-terminations) * self.discount_factor_g * atr_target_dqn(new_states).max(dim=1)[0]

        vtr_current_q = vtr_policy_dqn(states).gather(1, vtr_actions.unsqueeze(1)).squeeze(1)
        atr_current_q = atr_policy_dqn(states).gather(1, atr_actions.unsqueeze(1)).squeeze(1)

        vtr_loss = self.loss_fn(vtr_current_q, vtr_target_q)
        atr_loss = self.loss_fn(atr_current_q, atr_target_q)

        self.vtr_optimizer.zero_grad()
        self.atr_optimizer.zero_grad()
        vtr_loss.backward()
        atr_loss.backward()
        self.vtr_optimizer.step()
        self.atr_optimizer.step()


    def evaluate(self, num_episodes=10, render=False):
        env = gym.make(self.env_id, render_mode='human' if render else None)
        # 수정된 부분: action_space가 Tuple인 경우를 처리
        if isinstance(env.action_space, gym.spaces.Tuple):
            num_vtr_actions = env.action_space.spaces[0].n
            num_atr_actions = env.action_space.spaces[1].n
            print(f"VTR actions: {num_vtr_actions}, ATR actions: {num_atr_actions}")
        else:
            raise ValueError("Unexpected action space structure")


        model1_atr_path = 'D:\CodeRepository\AI_SCD 학습모델모음\환경I_1000EPI_길_env2\\eqp-scheduler-v0_atr.pt'
        model1_vtr_path = 'D:\CodeRepository\AI_SCD 학습모델모음\환경I_1000EPI_길_env2\\eqp-scheduler-v0_vtr.pt'

        model2_atr_path = 'D:\CodeRepository\AI_SCD 학습모델모음\환경I_1000EPI_짧_env3\\eqp-scheduler-v0_atr.pt'
        model2_vtr_path = 'D:\CodeRepository\AI_SCD 학습모델모음\환경I_1000EPI_짧_env3\\eqp-scheduler-v0_vtr.pt'

        model3_atr_path = 'D:\CodeRepository\AI_SCD 학습모델모음\환경I_1000EPI_짧_env2\\eqp-scheduler-v0_atr.pt'
        model3_vtr_path = 'D:\CodeRepository\AI_SCD 학습모델모음\환경I_1000EPI_짧_env2\\eqp-scheduler-v0_vtr.pt'

        model4_atr_path = 'D:\CodeRepository\AI_SCD 학습모델모음\Test\eqp-scheduler-v0_atr.pt'
        model4_vtr_path = 'D:\CodeRepository\AI_SCD 학습모델모음\Test\eqp-scheduler-v0_vtr.pt'

        # �н��� �� �ε�
        policy_dqn_model1_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model1_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model1_atr.load_state_dict(torch.load(model1_atr_path))
        policy_dqn_model1_vtr.load_state_dict(torch.load(model1_vtr_path))
        policy_dqn_model1_atr.eval()
        policy_dqn_model1_vtr.eval()

        policy_dqn_model2_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model2_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model2_atr.load_state_dict(torch.load(model2_atr_path))
        policy_dqn_model2_vtr.load_state_dict(torch.load(model2_vtr_path))
        policy_dqn_model2_atr.eval()
        policy_dqn_model2_vtr.eval()

        policy_dqn_model3_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model3_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model3_atr.load_state_dict(torch.load(model3_atr_path))
        policy_dqn_model3_vtr.load_state_dict(torch.load(model3_vtr_path))
        policy_dqn_model3_atr.eval()
        policy_dqn_model3_vtr.eval()   

        dummy_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        dummy_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)

        model1_wafers = []
        model2_wafers = []
        model3_wafers = []
        random_wafers = []
        optimal1_wafers = []
        optimal2_wafers = []

        for _ in range(num_episodes):
            model1_wafer = self.run_episode(env, policy_dqn_model1_atr, policy_dqn_model1_vtr, use_model=True)
            model2_wafer = self.run_episode(env, policy_dqn_model2_atr, policy_dqn_model2_vtr, use_model=True)
            model3_wafer = self.run_episode(env, policy_dqn_model3_atr, policy_dqn_model3_vtr, use_model=True)
            random_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 0)
            optimal1_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 1)
            optimal2_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 2)
            
            model1_wafers.append(model1_wafer)
            model2_wafers.append(model2_wafer)
            model3_wafers.append(model3_wafer)
            random_wafers.append(random_wafer)
            optimal1_wafers.append(optimal1_wafer)
            optimal2_wafers.append(optimal2_wafer)

        env.close()
        return model1_wafers, model2_wafers, model3_wafers, random_wafers, optimal1_wafers, optimal2_wafers

    def run_episode(self, env, policy_dqn_atr, policy_dqn_vtr, use_model=True, rule_based_type = 0):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        
        done = False

        while not done:
            invalid_vtr_actions, invalid_atr_actions = env.invalid_actions()
            
            if use_model:
                with torch.no_grad():
                    vtr_q_values = policy_dqn_vtr(state.unsqueeze(dim=0)).squeeze()
                    atr_q_values = policy_dqn_atr(state.unsqueeze(dim=0)).squeeze()
                    vtr_q_values[invalid_vtr_actions] = float('-inf')
                    atr_q_values[invalid_atr_actions] = float('-inf')
                    vtr_action = vtr_q_values.argmax().item()
                    atr_action = atr_q_values.argmax().item()
            else:
                # valid random
                if rule_based_type == 0:
                    vtr_action = random.choice([a for a in range(env.action_space.spaces[0].n) if a not in invalid_vtr_actions])
                    atr_action = random.choice([a for a in range(env.action_space.spaces[1].n) if a not in invalid_atr_actions])
                # optimal 1
                elif rule_based_type == 1:
                    
                    vtr_action_space_size = env.action_space.spaces[0].n
                    if set(range(1, vtr_action_space_size)) - set(invalid_vtr_actions):
                        invalid_vtr_actions = list(set(invalid_vtr_actions) | {0})

                    atr_action_space_size = env.action_space.spaces[1].n
                    if set(range(1, atr_action_space_size)) - set(invalid_atr_actions):
                        invalid_atr_actions = list(set(invalid_atr_actions) | {0})


                    vtr_action = random.choice([a for a in range(env.action_space.spaces[0].n) if a not in invalid_vtr_actions])
                    atr_action = random.choice([a for a in range(env.action_space.spaces[1].n) if a not in invalid_atr_actions])

                # optimal 2
                elif rule_based_type == 2:
                    # Optimal Policy (로봇 이동시간 고려)
                    vtr_action_space_size = env.action_space.spaces[0].n
                    if set(range(1, vtr_action_space_size)) - set(invalid_vtr_actions):
                        invalid_vtr_actions = list(set(invalid_vtr_actions) | {0})

                    atr_action_space_size = env.action_space.spaces[1].n
                    if set(range(1, atr_action_space_size)) - set(invalid_atr_actions):
                        invalid_atr_actions = list(set(invalid_atr_actions) | {0})

                    valid_vtr_actions = [a for a in range(vtr_action_space_size) if a not in invalid_vtr_actions]
                    valid_atr_actions = [a for a in range(atr_action_space_size) if a not in invalid_atr_actions]
                    
                    vtr_action = env.get_min_exe_time_vtr_action(valid_vtr_actions)
                    atr_action = env.get_min_exe_time_atr_action(valid_atr_actions)
                

            action = (vtr_action, atr_action)
            state, _, done, _, info = env.step(action)
            state = torch.tensor(state, dtype=torch.float, device=device)

        return info['total_wafer_processed']

    def compare_and_plot(self, num_episodes=10):
        model_wafers, model2_wafers, model3_wafers, random_wafers, optimal1_wafers, optimal2_wafers = self.evaluate(num_episodes)
        # model_wafers, random_wafers = self.evaluate(num_episodes)

        # plt.figure(figsize=(10, 6))
        # sns.boxplot(data=[model_wafers, model2_wafers, model3_wafers, random_wafers], width=0.6)
        # plt.xticks([0, 1, 2, 3], ['Trained Model 1', 'Trained Model 2', 'Trained Model 3', 'Random'])
        # plt.ylabel('Total Wafers Processed per Episode')
        # plt.title('Comparison of Trained Model vs Random Policy')
        # plt.savefig('model_vs_random_comparison.png')
        # plt.close()

        print(f"Trained Model - Mean: {np.mean(model_wafers):.2f}, Std: {np.std(model_wafers):.2f}")
        print(f"Trained Model 2 - Mean: {np.mean(model2_wafers):.2f}, Std: {np.std(model2_wafers):.2f}")
        print(f"Trained Model 3 - Mean: {np.mean(model3_wafers):.2f}, Std: {np.std(model3_wafers):.2f}")
        print(f"Random Sampling - Mean: {np.mean(random_wafers):.2f}, Std: {np.std(random_wafers):.2f}")
        print(f"Optimal 1 - Mean: {np.mean(optimal1_wafers):.2f}, Std: {np.std(optimal1_wafers):.2f}")
        print(f"Optimal 2 - Mean: {np.mean(optimal2_wafers):.2f}, Std: {np.std(optimal2_wafers):.2f}")

        

if __name__ == '__main__':
    dql = Agent()
    #dql.run(is_training=True)
    #dql.run(trained_model_file='20240826-1926-eqp-scheduler-v0', is_training=False, render=True)
    dql.compare_and_plot(num_episodes=1)