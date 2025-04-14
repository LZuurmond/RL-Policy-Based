import pandas as pd
import matplotlib.pyplot as plt
import os

def smooth(data, weight=0.9):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def load_and_smooth_rewards(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    df = pd.read_csv(filename)
    #rewards = df["Average Reward (past 100)"].tolist()
    rewards = df["Average Reward (past 100)"].tolist()
    rewards = smooth(rewards, weight=0.9)
    return df["Episode"], rewards

def plot(lr):
    methods = {
        "REINFORCE": f"training_rewards_reinforce_{str(lr)[2:]}.csv",
        "Actor-Critic": f"training_rewards_AC_{str(lr)[2:]}.csv",
        "A2C": f"training_rewards_A2C_{str(lr)[2:]}.csv"
    }

    plt.figure(figsize=(10, 6))

    for label, filepath in methods.items():
        episodes, smoothed_rewards = load_and_smooth_rewards(filepath)
        if episodes is not None:
            plt.plot(episodes, smoothed_rewards, label=label)

    plt.title("Training Performance (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"training_comparison_{str(lr)[2:]}.png")  # optional: save the plot
    plt.show()

if __name__ == "__main__":
    plot(0.001)