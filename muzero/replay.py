import base64
import json
import gzip
import os
from typing import List, NamedTuple, Tuple
import numpy as np
from torch import Tensor
import torch
from threading import Lock

from muzero.game import Game
import shutil


class StateIndex(NamedTuple):
    """Index of a single state in the replay buffer."""

    game_index: int
    state_index: int


class BatchedExperiences(NamedTuple):
    """A batch of experiences sampled from the replay buffer."""

    indexes: List[StateIndex]
    states: Tensor
    gradient_scales: Tensor
    actions: Tensor
    values: Tensor
    rewards: Tensor
    policy_probabilities: Tensor
    corrections: Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        unroll_steps: int,
        td_steps: int,
        path: str,
    ):
        """
        Initialize the replay buffer.

        Parameters
        ----------
        capacity: int
            The maximum number of experiences that the replay buffer can store.
        unroll_steps: int
            The number of steps to unroll the MCTS search.
        td_steps: int
            The number of steps to bootstrap the value target.
        path: str
            The path to the file where the replay buffer is stored.
        """
        self._capacity = capacity
        self._unroll_steps = unroll_steps
        self._td_steps = td_steps

        self._buffer: List[Game] = [None] * capacity
        self._priorities = np.zeros(capacity)
        self._reanalyze_priorities = np.zeros(capacity)

        # Lock to ensure that the buffer is thread-safe across
        # multiple actors, reanalyzer and the trainer.
        self._lock = Lock()
        self.total_games = 0
        self.total_samples = 0

        self._path = path
        self.load_from_disk()

    def save(self, game: Game):
        for i in range(len(game.states) - 1):
            priority = game.get_state_initial_priority(i, self._td_steps)
            game.priorities.append(priority)

        game.priorities = np.array(game.priorities)

        index = self.total_games % self._capacity

        with self._lock:
            self._buffer[index] = game
            self._priorities[index] = np.max(game.priorities)
            self._reanalyze_priorities[index] = 1
            self.total_games += 1
            self.total_samples += len(game.root_values)

    def sample(
        self,
        batch_size: int,
        device: str,
        n_states_representation: int,
        n_actions_representation: int,
    ) -> BatchedExperiences:
        selected_indexes = self._sample_indexes(batch_size)

        with self._lock:
            total_samples = self.total_samples
            game_data: List[Tuple[float, Game, int, int]] = [
                (
                    self._priorities[i],
                    self._buffer[i],
                    self._get_absolute_game_index(i),
                    self._sample_game_position(self._buffer[i]),
                )
                for i in selected_indexes
            ]

        states = []
        gradient_scales = []
        actions = []
        values = []
        rewards = []
        policy_probabilities = []
        corrections = []

        for game_priority, game, _, state_index in game_data:
            states.append(
                game.get_state(
                    state_index,
                    n_states_representation,
                    n_actions_representation,
                ).squeeze()
            )
            target_actions = game.get_action_history()[
                state_index : state_index + self._unroll_steps
            ]
            target_values, target_rewards, target_policy = zip(
                *game.get_targets(state_index, self._unroll_steps, self._td_steps)
            )

            gradient_scales.append(1.0 / len(target_actions))

            if len(target_actions) < self._unroll_steps:
                target_actions += [0] * (self._unroll_steps - len(target_actions))

            actions.append(target_actions)
            values.append(target_values)
            rewards.append(target_rewards)
            policy_probabilities.append(target_policy)
            corrections.append(
                1 / (total_samples * game_priority * game.priorities[state_index])
            )

        return BatchedExperiences(
            indexes=[
                StateIndex(game_index, state_index)
                for _, _, game_index, state_index in game_data
            ],
            states=torch.stack(states).to(device),
            gradient_scales=Tensor(gradient_scales).to(device),
            actions=Tensor(actions).to(device),
            values=Tensor(values).to(device),
            rewards=Tensor(rewards).to(device),
            policy_probabilities=Tensor(policy_probabilities).to(device),
            corrections=Tensor(corrections).to(device) / max(corrections),
        )

    def _sample_indexes(self, batch_size: int):
        selected_indexes = np.random.choice(
            self._capacity,
            batch_size,
            p=self._priorities / np.sum(self._priorities),
        )

        return selected_indexes

    def _sample_game_position(self, game: Game):
        return np.random.choice(
            len(game.states) - 1,
            p=game.priorities / np.sum(game.priorities),
        )

    def to_dict(self):
        return {
            "capacity": self._capacity,
            "buffer": self._buffer,
            "total_games": self.total_games,
            "total_samples": self.total_samples,
            "reanalyze_priorities": self._reanalyze_priorities,
        }

    def save_to_disk(self):
        if os.path.exists(self._path):
            old_path = self._get_old_path()
            shutil.copyfile(self._path, old_path)

        with gzip.open(self._path, "wt", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, cls=TensorJSONEncoder)

        if os.path.exists(self._get_old_path()):
            os.remove(old_path)

    def _get_old_path(self):
        return f"{self._path}.old"

    def load_from_disk(self):
        if not os.path.exists(self._path):
            return

        old_path = self._get_old_path()

        if os.path.exists(old_path):
            os.remove(self._path)
            os.rename(old_path, self._path)

        with gzip.open(self._path, "rt", encoding="utf-8") as f:
            data = json.load(f, object_hook=tensor_json_decoder)
            self._capacity = data["capacity"]
            self._buffer = data["buffer"]
            self.total_games = data["total_games"]
            self.total_samples = data["total_samples"]
            self._reanalyze_priorities = np.array(
                data.get("reanalyze_priorities", np.zeros(self._capacity))
            )
            self._reanalyze_priorities[0] = 1
            self._priorities = np.array(
                [
                    np.max(game.priorities) if game is not None else 0
                    for game in self._buffer
                ]
            )

    def update_priorities(self, indexes: List[StateIndex], priorities: np.ndarray):
        with self._lock:
            for (game_index, first_index), game_priorities in zip(indexes, priorities):
                if self._has_game_beed_replaced(game_index):
                    continue

                game_index = self._get_relative_game_index(game_index)
                game = self._buffer[game_index]

                priorities_last_index = min(
                    first_index + self._unroll_steps,
                    len(game.priorities),
                )
                new_priorities_last_index = priorities_last_index - first_index
                game.priorities[first_index:priorities_last_index] = game_priorities[
                    :new_priorities_last_index
                ]
                self._priorities[game_index] = np.max(game.priorities)

    def sample_game_for_reanalyze(self) -> Tuple[int, Game]:
        with self._lock:
            index = np.random.choice(
                self._capacity,
                p=self._reanalyze_priorities / np.sum(self._reanalyze_priorities),
            )
            self._reanalyze_priorities[index] = 0
            return self._get_absolute_game_index(index), self._buffer[index]

    def update_game(self, index: int, game: Game):
        with self._lock:
            if self._has_game_beed_replaced(index):
                return

            index = self._get_relative_game_index(index)
            self._buffer[index] = game
            self._priorities[index] = np.max(game.priorities)
            self._reanalyze_priorities[: min(self.total_games, self._capacity)] += 1

    def _get_absolute_game_index(self, game_index: int) -> int:
        """Absolute game index is used because the buffer is cyclic and the game may be replaced."""

        absolute_game_index = (
            self.total_games // self._capacity * self._capacity + game_index
        )

        if self.total_games % self._capacity <= game_index:
            absolute_game_index -= self._capacity

        return absolute_game_index

    def _get_relative_game_index(self, game_index: int) -> int:
        return game_index % self._capacity

    def _has_game_beed_replaced(self, game_index: int) -> bool:
        relative_game_index = self._get_relative_game_index(game_index)
        return self._get_absolute_game_index(relative_game_index) != game_index


class TensorJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            tensor_data = {
                "type": "torch.Tensor",
                "shape": obj.shape,
                "dtype": str(obj.dtype).split(".")[-1],
                "device": str(obj.device),
                "data": base64.b64encode(obj.numpy().tobytes()).decode("utf-8"),
            }
            return tensor_data
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Game):
            return obj.to_dict()
        return super().default(obj)


loaded_games = 0


def tensor_json_decoder(obj):
    global loaded_games
    if "type" in obj:
        if obj["type"] == "torch.Tensor":
            tensor_data = base64.b64decode(obj["data"])
            tensor = torch.frombuffer(tensor_data, dtype=getattr(torch, obj["dtype"]))
            return tensor.reshape(obj["shape"]).to(obj["device"])
        elif obj["type"] == "Game":
            loaded_games += 1
            print(f"Loaded {loaded_games} games...")

            return Game.from_dict(obj)
    return obj
