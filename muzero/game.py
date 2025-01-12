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

        self.priorities = []

        if env is not None:
            state, *_ = env.reset()
            self.states.append(self._state_to_tensor(state))

    def terminal(self) -> bool:
        return self.done

    def get_state(
        self,
        index: int,
        n_states_representation: int,
        n_actions_representation: int,
    ) -> torch.Tensor:
        stacked_states = self.states[
            max(index + 1 - n_states_representation, 0) : index + 1
        ]

        if not stacked_states:
            print(len(self.states), index, n_states_representation)

        if len(stacked_states) < n_states_representation:
            stacked_states = [torch.zeros_like(stacked_states[0])] * (
                n_states_representation - len(stacked_states)
            ) + stacked_states

        stacked_states = torch.cat(stacked_states, dim=1).float() / 255.0

        picked_actions = self.actions[
            max(index + 1 - n_actions_representation, 0) : index + 1
        ]
        if len(picked_actions) < n_actions_representation:
            picked_actions = [0] * (
                n_actions_representation - len(picked_actions)
            ) + picked_actions

        stacked_actions = (
            torch.tensor(picked_actions)
            .view(-1, *np.ones(len(stacked_states.shape[2:])).astype(int))
            .expand(-1, *stacked_states.shape[2:])
            .unsqueeze(0)
            .float()
        ) / 18.0

        return torch.cat([stacked_states, stacked_actions], dim=1)

    def get_action_history(self):
        return self.actions

    def get_state_initial_priority(self, state_index: int, td_steps: int):
        bootstrap_index = state_index + td_steps
        value = 0

        for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
            value += reward * self.discount**i

        if bootstrap_index < len(self.root_values):
            value += self.root_values[bootstrap_index] * self.discount**td_steps

        return abs(self.root_values[state_index] - value)

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

    def clone_for_reanalyze(self):
        game = Game(
            env=None,
            action_space_size=self.action_space_size,
            discount=self.discount,
            device=self.device,
        )

        game.states = self.states
        game.actions = self.actions
        game.rewards = self.rewards
        game.done = self.done
        return game

    def to_dict(self):
        return {
            "type": "Game",
            "states": self.states,
            "actions": self.actions,
            "discount": self.discount,
            "rewards": self.rewards,
            "done": self.done,
            "child_visits": self.child_visits,
            "root_values": self.root_values,
            "priorities": self.priorities,
            "action_space_size": self.action_space_size,
            "device": self.device,
        }

    @staticmethod
    def from_dict(data):
        game = Game(
            env=None,
            action_space_size=data["action_space_size"],
            discount=data["discount"],
            device=data["device"],
        )

        game.states = data["states"]
        game.actions = data["actions"]
        game.discount = data["discount"]
        game.rewards = data["rewards"]
        game.done = data["done"]
        game.child_visits = data["child_visits"]
        game.root_values = data["root_values"]
        game.priorities = data["priorities"]
        return game
