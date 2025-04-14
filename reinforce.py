import argparse
import time
import numpy as np
import pandas as pd
import gym
import torch
from torch import nn, optim


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.0005)
    parser.add_argument("--gamma", type=float, help="Discount Factor", default=0.99)
    parser.add_argument("--hidden_dim", type=int, help="Size of hidden layer in network", default=128)
    parser.add_argument("--num_episodes", type=int, help="Number of Episodes to train on", default=1000)
    parser.add_argument("--no_cuda", type=bool, help="Set to true if not using cuda (when available)", default=False)
    parser.add_argument("--seed", type=int, help="Set a seed for reproducibility", default=int(time.time()))

    return parser


def create_model(state_dim, action_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
        nn.Softmax(dim=-1)
    )


def run_experiment(lr, gamma, layers, num_episodes, no_cuda):
    # Check if cuda is available and use if it is
    device = torch.device("cuda" if not no_cuda and torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create Network
    model = create_model(state_dim, action_dim, hidden_dim).to(device)

    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    training_data = []
    episode_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float).to(device)
        done = False
        Actions, States, Rewards = [], [], []

        # run the episode
        while not done:
            probs = model(state)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()
            next_state, reward, done1, done2, _ = env.step(action)
            done = done1 or done2

            Actions.append(action)
            States.append(state)
            Rewards.append(reward)

            state = torch.tensor(next_state, dtype=torch.float).to(device)

        # transform taken actions into a tensor
        Actions = torch.tensor(Actions, dtype=torch.int).to(device)

        # Compute discounted returns
        DiscountedReturns = []
        for t in range(len(Rewards)):
            G = 0.0
            for k,r in enumerate(Rewards[t:]):
                G += (gamma ** k) * r
            DiscountedReturns.append(G)
        Returns = torch.tensor(DiscountedReturns, dtype=torch.float).to(device)
        Returns = (Returns - Returns.mean()) / (Returns.std() + 1e-8)  # normalize the returns

        # Update gradients
        episode_loss = 0.0
        for State, Action, G in zip(States, Actions, Returns):
            probs = model(State)
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(Action)

            episode_loss += - log_prob * G

        optimizer.zero_grad()
        episode_loss.backward()
        optimizer.step()

        # Log episode return
        episode_reward = sum(Rewards)
        episode_rewards.append(episode_reward)
        training_data.append([episode + 1, episode_reward, np.mean(episode_rewards[-100:])])
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Average Reward: {np.mean(episode_rewards[-100:])}")

    return model, training_data

if __name__ == '__main__':
    parser = create_argparse()
    args = parser.parse_args()

    lr = args.lr
    gamma = args.gamma
    hidden_dim = args.hidden_dim
    num_episodes = args.num_episodes
    no_cuda = args.no_cuda
    seed = args.seed

    torch.manual_seed(seed)

    model, training_data = run_experiment(lr, gamma, hidden_dim, num_episodes, no_cuda)

    # store training data
    df = pd.DataFrame(training_data, columns=["Episode", "Episode Reward", "Average Reward (past 100)"])
    df.to_csv(f"training_rewards_reinforce_{str(lr)[2:]}.csv", index=False)