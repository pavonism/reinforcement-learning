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
    def __init__(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones


class ReplayBuffer:
    def __init__(
        self, max_reward_value: float, capacity: int = 1_000_000, seed: int = 0
    ):
        self.buffer = deque(maxlen=capacity)
        self.max_reward_value = max_reward_value
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

    def sample_experience(
        self,
        device,
        state_memory_batch_size: int,
        batch_size: int = 32,
    ) -> TensorBatchExperience:
        experiences_indexes = random.choices(
            range(len(self.buffer)),
            k=batch_size,
        )

        experiences = self.__gather_experiences(
            experiences_indexes,
            state_memory_batch_size,
        )

        # Separating each component to make batch processing easier
        states, actions, rewards, next_states, dones = zip(*experiences)

        return TensorBatchExperience(
            torch.stack(states).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32)
            .div(self.max_reward_value)
            .to(device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device),
        )

    def __gather_experiences(self, indexes: list, state_memory_batch_size: int):
        experiences = []

        for index in indexes:
            current_experiences = [
                self.buffer[i]
                for i in range(max(0, index - state_memory_batch_size + 1), index + 1)
            ]

            states, actions, rewards, next_states, dones = zip(*current_experiences)

            # Filling the states with the first state if there is not enough states
            if len(states) < state_memory_batch_size:
                states = self.__expand_states_to_batch_size(
                    state_memory_batch_size, states
                )

            if len(next_states) < state_memory_batch_size:
                next_states = self.__expand_states_to_batch_size(
                    state_memory_batch_size, next_states
                )

            experiences.append(
                (
                    torch.stack(states),
                    actions[-1],
                    rewards[-1],
                    torch.stack(next_states),
                    dones[-1],
                )
            )

        return experiences

    def __expand_states_to_batch_size(self, state_memory_batch_size, states):
        states = [states[0]] * (state_memory_batch_size - len(states)) + list(states)

        return states

    def get_last_states(self, count: int) -> torch.Tensor:
        experiences = [
            self.buffer[index]
            for index in range(max(0, len(self.buffer) - count), len(self.buffer))
        ]

        # Separating each component to make batch processing easier
        _, _, _, next_states, _ = zip(*experiences)

        next_states = self.__expand_states_to_batch_size(count, next_states)

        return torch.stack(next_states)

    def save(self, path):
        torch.save(self.buffer, f"{path}/replay_buffer.pt")

    def load(self, path):
        self.buffer = torch.load(f"{path}/replay_buffer.pt", weights_only=False)

    def __len__(self):
        return len(self.buffer)
