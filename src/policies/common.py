from collections import deque
import random

import numpy as np
import torch
from PIL import Image


class Policy:
    def train(self, episodes: int, max_steps: int) -> float:
        pass

    def play(self, episodes: int, max_steps: int) -> float:
        pass


class TensorBatchExperience:
    def __init__(self, states, actions, rewards, next_states, dones, device):
        self.states = states
        self.actions = torch.tensor(actions, dtype=torch.long).to(device)
        self.rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        self.next_states = next_states
        self.dones = torch.tensor(dones, dtype=torch.float32).to(device)


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000, seed: int = 0):
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)

    def remember_experience(
        self,
        state: Image.Image,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_experience(self, device, batch_size: int = 32) -> TensorBatchExperience:
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Separating each component to make batch processing easier
        states, actions, rewards, next_states, dones = zip(*experiences)

        return TensorBatchExperience(
            list(states),
            actions,
            rewards,
            list(next_states),
            dones,
            device=device,
        )

    def __len__(self):
        return len(self.buffer)
