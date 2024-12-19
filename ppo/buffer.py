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
        
        rewards_mean = rewards.mean()
        rewards_std = rewards.std() + 1e-8  # Avoid division by zero
        rewards = (rewards - rewards_mean) / rewards_std

        return rewards
