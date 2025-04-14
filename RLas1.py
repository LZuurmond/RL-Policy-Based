import gym
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


def run_experiment(param_name):
    results = {}

    # default parameters
    lr = 0.0001
    hidden_dim = 128
    buffer_size = 100000
    num_episodes = 3000
    epsilon = 1.0  # exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99  # discount factor
    batch_size = 64
    target_update_freq = 10

    for value in hyperparameters[param_name]:
        # Set current hyperparameter while keeping others at medium values
        if param_name == "learning_rate":
            lr = value
        elif param_name == "network_size":
            hidden_dim = value
        elif param_name == "update_to_data_ratio":
            update_to_data_ratio = value
        elif param_name == "exploration_factor":
            epsilon = value

        # Initialize environment
        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Initialize Q-network and target network
        dqn = DQN(state_dim, hidden_dim, action_dim)
        target_dqn = DQN(state_dim, hidden_dim, action_dim)
        target_dqn.load_state_dict(dqn.state_dict())

        # Define optimizer and loss function
        optimizer = optim.Adam(dqn.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # Initialize replay buffer
        buffer = ReplayBuffer(buffer_size)

        data = []

        # Training Loop
        for episode in range(num_episodes):
            state = env.reset()[0]
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < env._max_episode_steps:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action = torch.argmax(dqn(state_tensor)).item()

                # take action in the environment
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                step_count += 1

                # store transition in the replay buffer
                buffer.push(state, action, reward, next_state, done)
                state = next_state

                # training step
                if len(buffer) > batch_size:
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions)
                    rewards = torch.FloatTensor(rewards)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones)

                    # compute target q-values
                    with torch.no_grad():
                        target_q_values = rewards + gamma * (1 - dones) * torch.max(target_dqn(next_states), dim=1)[0]

                    # compute current q-values
                    q_values = dqn(states).gather(1,actions.unsqueeze(1)).squeeze(1)

                    # compute loss
                    loss = loss_fn(q_values, target_q_values)

                    # backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # update target network
            if episode % target_update_freq == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            # decay epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            data.append([episode + 1, episode_reward])
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}")

        results[value] = data

    return results


# hyperparameters
hyperparameters = {
    "lr": [0.001, 0.0001, 0.00001],
    "target_update_freq": [5, 10, 20],
    "hidden_dim": [64, 128, 256],
    "epsilon_decay": [0.99, 0.995, 0.999]
}

# save results
learning_data = run_experiment("hidden_dim")
learning_data_list = []
for i in range(3000):
    learning_data_list.append([i+1])
for value in learning_data:
    for episode, episode_return in learning_data[value]:
        learning_data_list[episode-1].append(episode_return)

print(learning_data_list)

learning_data = pd.DataFrame(learning_data_list, columns=["Episode", "64", "128", "256"])
learning_data.to_csv("ablation_network_size.csv", index=False)