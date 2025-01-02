import numpy as np
import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def compute_rewards_to_go(self, gamma=0.99):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        return self.normalize_rewards(rewards)

    
    def compute_gae(self, values, gamma=0.99, gae_lambda=0.95):
        advantages = []
        gae = 0
        next_value = 0

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * next_value * (1 - self.is_terminals[step]) - values[step]
            gae = delta + gamma * gae_lambda * gae * (1 - self.is_terminals[step])
            advantages.insert(0, gae)
            next_value = values[step]

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def normalize_rewards(self, rewards):
        rewards_mean = rewards.mean()
        rewards_std = rewards.std() + 1e-8
        return (rewards - rewards_mean) / rewards_std

