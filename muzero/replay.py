import pickle
import gzip
from typing import List, NamedTuple, Tuple
import numpy as np

from muzero.game import Game


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    policy_probabilities: np.ndarray
    value: float
    reward: float


class BatchedExperiences(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    policy_probabilities: np.ndarray
    values: np.ndarray
    rewards: np.ndarray


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

    def sample(self, steps: int, td_steps: int, batch_size: int) -> BatchedExperiences:
        selected_indexes = np.random.choice(
            min(self._capacity, self.total_games),
            min(batch_size, self.total_games),
            replace=False,
        )

        game_pos: List[Tuple[Game, int]] = [
            (self._buffer[i], self._sample_game_position(self._buffer[i]))
            for i in selected_indexes
        ]

        return [
            (
                g.get_state(state_index),
                g.get_action_history()[state_index : state_index + steps],
                g.get_targets(state_index, steps, td_steps),
            )
            for (g, state_index) in game_pos
        ]

    def _sample_game_position(self, game: Game):
        return np.random.choice(len(game.states))

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
