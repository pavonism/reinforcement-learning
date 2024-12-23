import pickle
import gzip
from typing import List, NamedTuple, Tuple
import numpy as np
from torch import Tensor
import torch

from muzero.game import Game


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    policy_probabilities: np.ndarray
    value: float
    reward: float


class BatchedExperiences(NamedTuple):
    states: Tensor
    gradient_scales: Tensor
    actions: Tensor
    values: Tensor
    rewards: Tensor
    policy_probabilities: Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
    ):
        """
        Initialize the replay buffer.

        Parameters
        ----------
        capacity: int
            The maximum number of experiences that the replay buffer can store.
        """
        self._capacity = capacity

        self._buffer: List[Game] = [None] * capacity
        self.total_games = 0

    def save(self, game: Game):
        index = self.total_games % self._capacity
        self._buffer[index] = game
        self.total_games += 1

    def sample(
        self,
        steps: int,
        td_steps: int,
        batch_size: int,
        device: str,
    ) -> BatchedExperiences:
        selected_indexes = np.random.choice(
            min(self._capacity, self.total_games),
            batch_size,
        )

        game_pos: List[Tuple[Game, int]] = [
            (self._buffer[i], self._sample_game_position(self._buffer[i]))
            for i in selected_indexes
        ]

        states = []
        gradient_scales = []
        actions = []
        values = []
        rewards = []
        policy_probabilities = []

        for g, state_index in game_pos:
            states.append(g.get_state(state_index).squeeze())
            target_actions = g.get_action_history()[state_index : state_index + steps]
            target_values, target_rewards, target_policy = zip(
                *g.get_targets(state_index, steps, td_steps)
            )

            gradient_scales.append(1.0 / len(target_actions))

            if len(target_actions) < steps:
                target_actions += [0] * (steps - len(target_actions))

            actions.append(target_actions)
            values.append(target_values)
            rewards.append(target_rewards)
            policy_probabilities.append(target_policy)

        return BatchedExperiences(
            states=torch.stack(states).to(device),
            gradient_scales=Tensor(gradient_scales).to(device),
            actions=Tensor(actions).to(device),
            values=Tensor(values).to(device),
            rewards=Tensor(rewards).to(device),
            policy_probabilities=Tensor(policy_probabilities).to(device),
        )

    def _sample_game_position(self, game: Game):
        return np.random.choice(len(game.states) - 1)

    def to_dict(self):
        return {
            "capacity": self._capacity,
            "buffer": self._buffer,
            "total_games": self.total_games,
        }

    def save_to_disk(self, path: str):
        with gzip.open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    def load_from_disk(self, path: str):
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
            self._capacity = data["capacity"]
            self._buffer = data["buffer"]
            self.total_games = data["total_games"]
