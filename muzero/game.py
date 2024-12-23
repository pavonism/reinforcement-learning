from typing import Dict
from gymnasium import Env
import numpy as np
import torch


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.probability = prior
        self.values_sum = 0
        self.children: Dict[float, Node] = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.values_sum / self.visit_count


class Game(object):
    def __init__(
        self,
        env: Env,
        action_space_size: int,
        discount: float,
        device: str,
    ):
        self.env = env
        self.action_space_size = action_space_size
        self.discount = discount
        self.device = device

        self.states = []
        self.actions = []
        self.rewards = []
        self.done = False

        self.child_visits = []
        self.root_values = []

        state, *_ = env.reset()
        self.states.append(self._state_to_tensor(state))

    def terminal(self) -> bool:
        return self.done

    def get_state(self, index: int) -> torch.Tensor:
        return self.states[index]

    def get_action_history(self):
        return self.actions

    def get_targets(
        self,
        state_index: int,
        steps: int,
        td_steps: int,
    ):
        targets = []
        for current_index in range(state_index, state_index + steps + 1):
            value = 0
            bootstrap_index = current_index + td_steps

            # Discounted sum of all rewards until the bootstrap_index.
            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i

            # Discounted root value of the search tree td_steps into the future.
            if bootstrap_index < len(self.root_values):
                value += self.root_values[bootstrap_index] * self.discount**td_steps

            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            last_reward = (
                self.rewards[current_index - 1]
                if current_index > 0 and current_index <= len(self.rewards)
                else 0
            )

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, [0] * self.action_space_size))

        return targets

    def apply_action(self, action):
        state, reward, done, *_ = self.env.step(action)
        self.states.append(self._state_to_tensor(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.done = done

    def store_search_statistics(self, root: Node):
        all_child_visits = sum(child.visit_count for child in root.children.values())
        action_space = range(self.action_space_size)
        self.child_visits.append(
            list(
                root.children[a].visit_count / all_child_visits
                if a in root.children
                else 0
                for a in action_space
            )
        )
        self.root_values.append(root.value())

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        state_tensor = torch.from_numpy(
            state
        ).float()  # Convert to tensor and float type
        state_tensor = state_tensor.permute(2, 0, 1)  # Reshape from HWC to CHW
        return state_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
