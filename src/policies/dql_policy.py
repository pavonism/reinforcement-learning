import copy
from gymnasium import Env
from torch import nn
import torch
from src.policies.common import Policy, ReplayBuffer
import numpy as np
from torchvision import transforms
from PIL import Image


class DQLPolicy(Policy):
    def __init__(
        self,
        env: Env,
        model: nn.Module,
        state_transformer: transforms.Compose = transforms.Compose(
            [
                transforms.Resize((80, 80)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        ),
        learning_rate: float = 0.001,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        discount_factor: float = 0.99,
        target_update_frequency: int = 10,
        seed: int = 0,
    ):
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__env = env
        self.__n_actions = env.action_space.n
        self.__action_value = model.to(self.__device)
        self.__target_action_value = copy.deepcopy(model)
        self.__state_transformer = state_transformer
        self.__optimizer = torch.optim.Adam(
            self.__action_value.parameters(),
            lr=learning_rate,
        )
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay
        self.__discount_factor = discount_factor
        self.__target_update_frequency = target_update_frequency
        self.__replay_buffer = ReplayBuffer(seed=seed)
        np.random.seed(seed)

    def run(self, episodes: int = 1, max_steps: int = 1_000) -> float:
        max_reward = 0

        for _ in range(episodes):
            state, _ = self.__env.reset()
            state = Image.fromarray(state)
            total_reward = 0
            target_update_counter = 0

            for step in range(max_steps):
                print("Step: ", step)
                action = self.__get_action_from_epsilon_greedy(state)
                new_state, reward, done, _, _ = self.__env.step(action)
                new_state = Image.fromarray(new_state)

                self.__replay_buffer.remember_experience(
                    state, action, reward, new_state, done
                )
                state = new_state

                experiences = self.__replay_buffer.sample_experience(self.__device)

                transformed_experience_states = torch.stack(
                    [self.__state_transformer(img) for img in experiences.states]
                ).to(self.__device)

                q_values = self.__action_value(transformed_experience_states).gather(
                    1, experiences.actions.view(-1, 1)
                )

                transformed_experience_next_states = torch.stack(
                    [self.__state_transformer(img) for img in experiences.next_states]
                ).to(self.__device)

                target_q_values = self.__target_action_value(
                    transformed_experience_next_states
                ).max(1)[0]

                y_i = (
                    experiences.rewards
                    + (1 - experiences.dones) * self.__discount_factor * target_q_values
                )

                loss = nn.MSELoss(reduction="none")
                output = loss(y_i, q_values.flatten()).mean()
                output.backward()
                self.__optimizer.step()

                total_reward += reward
                target_update_counter += 1

                if target_update_counter % self.__target_update_frequency == 0:
                    self.__target_action_value.load_state_dict(
                        self.__action_value.state_dict()
                    )

                if done:
                    break

            max_reward = max(max_reward, total_reward)
            self.__epsilon *= self.__epsilon_decay

        self.__env.reset()
        return max_reward

    def __get_action_from_epsilon_greedy(self, state):
        # Exploration
        if np.random.random() < self.__epsilon:
            return np.random.random_integers(0, self.__n_actions - 1, 1)[0]

        # Exploitation
        transformed_state = self.__state_transformer(state)
        transformed_state = transformed_state.unsqueeze(0)
        transformed_state = transformed_state.to(self.__device)

        self.__action_value.eval()
        with torch.no_grad():
            return self.__action_value(transformed_state).argmax().item()
