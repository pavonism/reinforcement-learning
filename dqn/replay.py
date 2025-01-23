import random
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image


class TensorBatchExperience(NamedTuple):
    """
    Structure to hold a batch of experiences in tensor format.
    """

    indexes: np.ndarray
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """
    Replay buffer to store experiences and sample them for DQN training.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        seed: int = 0,
        with_priorities: bool = True,
    ):
        """
        Replay buffer to store experiences and sample them for DQN training.

        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
            seed (int): Seed for random number generation.
            with_priorities (bool): Whether to use prioritized experience replay.
            min_priority (float): Minimum priority value for a sampled experience.
        """
        self.buffer = []
        self.priorities = torch.tensor(np.zeros(capacity)).cuda()
        self.position = 0
        self.with_priorities = with_priorities
        self.reward_normalizer = RewardNormalizer()
        self._random = random.Random(seed)

    def save(
        self,
        state: Image.Image,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Save an experience in the replay buffer.

        Args:
            state (Image.Image): Current state of the environment.
            action (int): Action taken in the current state.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Next state of the environment.
            done (bool): Whether the episode has finished.
        """
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < len(self.priorities):
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        max_priority = torch.max(self.priorities).item()
        self.priorities[self.position] = max_priority if max_priority > 0 else 1
        self.position = (self.position + 1) % len(self.priorities)

    def sample(
        self,
        device,
        stack_frame: int,
        batch_size: int = 32,
    ) -> TensorBatchExperience:
        """
        Sample a batch of experiences from the replay buffer.

        Args:
            device: Device to store the tensors.
            stack_frame (int): Number of states to consider for each experience.
            batch_size (int): Number of experiences to sample.
        """

        experience_indexes = self._sample_indexes(batch_size)

        experiences = self._gather_experiences(
            experience_indexes,
            stack_frame,
        )

        # Separating each component to make batch processing easier
        states, actions, rewards, next_states, dones = zip(*experiences)

        return TensorBatchExperience(
            indexes=experience_indexes,
            states=torch.stack(states).to(device),
            actions=torch.tensor(actions, dtype=torch.long).to(device),
            rewards=torch.tensor(
                self.reward_normalizer.normalize(rewards),
                dtype=torch.float32,
            ).to(device),
            next_states=torch.stack(next_states).to(device),
            dones=torch.tensor(dones, dtype=torch.float32).to(device),
        )

    def _sample_indexes(self, batch_size):
        # Sampling on GPU for faster processing
        experience_indexes = torch.multinomial(
            self.priorities,
            batch_size,
            replacement=len(self.buffer) < batch_size,
        ).cpu()

        return experience_indexes

    def _gather_experiences(self, indexes: list, stack_frame: int):
        """
        For each index, gather last stack_frame experiences from the buffer and stack them together.
        """
        experiences = []

        for index in indexes:
            current_experiences = [
                self.buffer[i]
                for i in range(max(0, index - stack_frame + 1), index + 1)
            ]

            states, actions, rewards, next_states, dones = zip(*current_experiences)

            # Filling the states with the first state if there is not enough states
            if len(states) < stack_frame:
                states = self._expand_states_to_batch_size(stack_frame, states)

            if len(next_states) < stack_frame:
                next_states = self._expand_states_to_batch_size(
                    stack_frame, next_states
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

    def _expand_states_to_batch_size(self, stack_frame: int, states: list):
        states = [states[0]] * (stack_frame - len(states)) + list(states)

        return states

    def get_last_states(self, count: int) -> torch.Tensor:
        experiences = [
            self.buffer[index]
            for index in range(max(0, len(self.buffer) - count), len(self.buffer))
        ]

        # Separating each component to make batch processing easier
        _, _, _, next_states, _ = zip(*experiences)

        next_states = self._expand_states_to_batch_size(count, next_states)

        return torch.stack(next_states)

    def update_priorities(self, indexes, priorities):
        if self.with_priorities:
            self.priorities[indexes] = priorities

    def save_to_directory(self, path: str):
        torch.save(self.buffer, f"{path}/replay_buffer.pt")
        torch.save(self.priorities, f"{path}/replay_buffer_priorities.pt")
        torch.save(self.reward_normalizer, f"{path}/reward_normalizer.pt")

    def load(self, path: str):
        self.buffer = torch.load(
            f"{path}/replay_buffer.pt",
            weights_only=False,
        )

        self.priorities = torch.load(
            f"{path}/replay_buffer_priorities.pt",
            weights_only=False,
        ).cuda()

        self.reward_normalizer = torch.load(
            f"{path}/reward_normalizer.pt",
            weights_only=False,
        )

    def __len__(self):
        return len(self.buffer)


class RewardNormalizer:
    """
    Normalizes rewards using running mean and variance.
    """

    def __init__(self, epsilon=1e-4):
        """
        Normalizes rewards using running mean and variance.

        Args:
            epsilon (float): Small value used for standard deviation calculation if the variance is small.
        """

        self.mean = 0
        self.var = 1
        self.count = 0
        self.epsilon = epsilon

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
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
