import random

import numpy as np
import torch
from PIL import Image

from line_profiler import profile


class Policy:
    def train(self, episodes: int, max_steps: int) -> float:
        pass

    def play(self, episodes: int, max_steps: int) -> float:
        pass


class TensorBatchExperience:
    def __init__(
        self,
        indexes: np.ndarray,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ):
        self.indexes = indexes
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 1_000_000,
        seed: int = 0,
        with_priorities: bool = True,
        min_priority: float = 0.01,
    ):
        self.buffer = []
        self.priorities = torch.tensor(np.zeros(capacity)).cuda()
        self.position = 0
        self.with_priorities = with_priorities
        self.__min_priority = min_priority
        self.reward_normalizer = RewardNormalizer()
        random.seed(seed)

    def remember_experience(
        self,
        state: Image.Image,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < len(self.priorities):
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.__min_priority
        self.position = (self.position + 1) % len(self.priorities)

    @profile
    def sample_experience(
        self,
        device,
        state_memory_batch_size: int,
        batch_size: int = 32,
    ) -> TensorBatchExperience:
        experience_indexes = torch.multinomial(
            self.priorities,
            batch_size,
            replacement=len(self.buffer) < batch_size,
        ).cpu()

        experiences = self.__gather_experiences(
            experience_indexes,
            state_memory_batch_size,
        )

        # Separating each component to make batch processing easier
        states, actions, rewards, next_states, dones = zip(*experiences)

        return TensorBatchExperience(
            experience_indexes,
            torch.stack(states).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(
                self.reward_normalizer.normalize(rewards),
                dtype=torch.float32,
            ).to(device),
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

    def update_priorities(self, indexes, priorities):
        if self.with_priorities:
            self.priorities[indexes] = priorities

    def save(self, path):
        torch.save(self.buffer, f"{path}/replay_buffer.pt")
        torch.save(self.priorities, f"{path}/replay_buffer_priorities.pt")
        torch.save(self.reward_normalizer, f"{path}/reward_normalizer.pt")

    def load(self, path):
        self.buffer = torch.load(f"{path}/replay_buffer.pt", weights_only=False)
        self.priorities = torch.load(
            f"{path}/replay_buffer_priorities.pt", weights_only=False
        ).cuda()
        self.reward_normalizer = torch.load(
            f"{path}/reward_normalizer.pt", weights_only=False
        )

    def __len__(self):
        return len(self.buffer)


class RewardNormalizer:
    def __init__(self, epsilon=1e-4):
        self.mean = 0
        self.var = 1
        self.count = 0
        self.epsilon = epsilon

    def normalize(self, rewards):
        rewards = np.array(rewards)

        new_count = self.count + len(rewards)
        new_mean = (self.count * self.mean + np.sum(rewards)) / new_count
        new_var = (
            self.count * self.var + np.sum((rewards - new_mean) ** 2)
        ) / new_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

        std = np.maximum(self.epsilon, np.sqrt(self.var))

        return (rewards - self.mean) / std
