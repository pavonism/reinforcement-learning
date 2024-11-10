from collections import deque
import os
import random

import numpy as np
import pandas as pd
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
        self.weights = []
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
        self.weights.append(reward)

    def sample_experience(self, device, batch_size: int = 32) -> TensorBatchExperience:
        experiences = random.choices(self.buffer, weights=self.weights, k=batch_size)

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

    def save(self, path: str):
        os.makedirs(f"{path}/replay_buffer", exist_ok=True)
        os.makedirs(f"{path}/replay_buffer/states", exist_ok=True)
        os.makedirs(f"{path}/replay_buffer/next_states", exist_ok=True)

        experiences = []

        for idx, experience in enumerate(self.buffer):
            state, action, reward, next_state, done = experience

            state_path = f"{path}/replay_buffer/states/{idx}.png"
            next_state_path = f"{path}/replay_buffer/next_states/{idx}.png"

            state.save(state_path)
            next_state.save(next_state_path)

            experiences.append(
                {
                    "state": state_path,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state_path,
                    "done": done,
                }
            )

        pd.DataFrame(experiences).to_csv(f"{path}/replay_buffer/experiences.csv")

    def load(self, path: str):
        experiences = pd.read_csv(f"{path}/replay_buffer/experiences.csv")

        for _, experience in experiences.iterrows():
            state = Image.open(experience["state"])
            next_state = Image.open(experience["next_state"])

            self.buffer.append(
                (
                    state,
                    experience["action"],
                    experience["reward"],
                    next_state,
                    experience["done"],
                )
            )

    def __len__(self):
        return len(self.buffer)
