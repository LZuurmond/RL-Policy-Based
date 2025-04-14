import argparse
import time
import gym
import torch
from torch import nn, optim
import pandas as pd
import numpy as np


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.0005)
    parser.add_argument("--gamma", type=float, help="Discount Factor", default=0.99)
    parser.add_argument("--hidden_dim", type=int, help="Hidden Layers of the network", default=128)
    parser.add_argument("--num_episodes", type=int, help="Number of Episodes to train on", default=1000)
    parser.add_argument("--no_cuda", type=bool, help="Set to true if not using cuda (when available)", default=False)
    parser.add_argument("--advantage", type=bool, help="Set to true if using A2C", default=False)
    parser.add_argument("--seed", type=int, help="Set a seed for reproducibility", default=int(time.time()))

    return parser


def create_models(state_dim, action_dim, hidden_dim, device=torch.device("cpu")):
    actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)

    critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    return actor, critic


def run_experiment(lr, gamma, hidden_dim, num_episodes, no_cuda, use_advantage):
    # Check if cuda is available and use if it is
    device = torch.device("cuda" if not no_cuda and torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create Network
    actor, critic = create_models(state_dim, action_dim, hidden_dim, device=device)

    # Optimizer and Loss Function
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    # Store episode rewards
    training_data = []
    episode_rewards = []


    for episode in range(num_episodes):
        episode_reward = 0
        done = False
        states, actions, rewards, log_probs = [], [], [], []

        # Reset environment
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float).to(device)

        # Collect a full episode
        while not done:
            # Sample action from model
            action_probs = actor(state)
            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # take step
            next_state, reward, done1, done2, _ = env.step(action.item())
            done = done1 or done2

            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            log_probs.append(log_prob)

            state = next_state
            episode_reward += reward

        # Compute returns in a single loop
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Compute values
        states = torch.stack(states)
        actions = torch.stack(actions).to(dtype=torch.long)
        log_probs = torch.stack(log_probs)
        values = critic(states)

        if use_advantage:
            advantages = returns - values.detach()
        else:
            advantages = returns

        # update critic
        critic_loss = nn.MSELoss()(values.squeeze(1), returns)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update actor
        actor_loss = - (log_probs * advantages).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # add episode reward to training data
        episode_rewards.append((episode_reward))
        training_data.append([episode + 1, episode_reward, np.mean(episode_rewards[-100:])])
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Average Reward: {np.mean(episode_rewards[-100:])}")

    return actor, critic, training_data


if __name__ == '__main__':
    parser = create_argparse()
    args = parser.parse_args()

    lr = args.lr
    gamma = args.gamma
    hidden_dim = args.hidden_dim
    num_episodes = args.num_episodes
    no_cuda = args.no_cuda
    advantage = args.advantage
    seed = args.seed

    torch.manual_seed(seed)

    _, _, training_data = run_experiment(lr, gamma, hidden_dim, num_episodes, no_cuda, advantage)

    # store training data
    df = pd.DataFrame(training_data, columns=["Episode", "Episode Reward", "Average Reward (past 100)"])
    df.to_csv(f"training_rewards_AC_{str(lr)[2:]}.csv", index=False)


    advantage = True
    _, _, training_data = run_experiment(lr, gamma, hidden_dim, num_episodes, no_cuda, advantage)
    # store training data
    df = pd.DataFrame(training_data, columns=["Episode", "Episode Reward", "Average Reward (past 100)"])
    df.to_csv(f"training_rewards_A2C_{str(lr)[2:]}.csv", index=False)