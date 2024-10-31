import copy
import json
import os
from gymnasium import Env
from torch import nn
import torch
from tqdm import tqdm
from src.policies.common import Policy, ReplayBuffer
import numpy as np
from torchvision import transforms
from PIL import Image


class DQLPolicy(Policy):
    def __init__(
        self,
        env: Env,
        model: nn.Module,
        path: str,
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
        save_frequency_in_episodes: int = 100,
        seed: int = 0,
    ):
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__env = env
        self.__n_actions = env.action_space.n
        self.__action_value = model.to(self.__device)
        self.__target_action_value = copy.deepcopy(model)
        self.__path = path
        self.__metrics_file = f"{path}/metrics.csv"
        self.__state_transformer = state_transformer
        self.__optimizer = torch.optim.Adam(
            self.__action_value.parameters(),
            lr=learning_rate,
        )
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay
        self.__discount_factor = discount_factor
        self.__target_update_frequency = target_update_frequency
        self.__save_frequency_in_episodes = save_frequency_in_episodes
        self.__replay_buffer = ReplayBuffer(seed=seed)
        self.__total_episodes = 0
        np.random.seed(seed)

        if os.path.exists(self.__path):
            self.__load()

    def train(self, episodes: int = 1, max_steps: int = 1_000) -> float:
        max_reward = 0

        for episode in range(self.__total_episodes, self.__total_episodes + episodes):
            state, _ = self.__env.reset()
            state = Image.fromarray(state)
            episode_total_reward = 0
            target_update_counter = 0

            for step in range(max_steps):
                action, action_type = self.__get_action_from_epsilon_greedy(state)
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

                episode_total_reward += reward
                target_update_counter += 1

                if target_update_counter % self.__target_update_frequency == 0:
                    self.__target_action_value.load_state_dict(
                        self.__action_value.state_dict()
                    )

                self.__log(episode, step, action, action_type, reward, output.item())

                if done:
                    break

            print(f"Total reward for episode {episode}: {episode_total_reward}")
            max_reward = max(max_reward, episode_total_reward)
            self.__epsilon *= self.__epsilon_decay
            self.__total_episodes += 1

            if episode % self.__save_frequency_in_episodes == 0:
                self.__save()

        self.__env.reset()
        self.__save()
        return max_reward

    def play(self, episodes: int = 1, max_steps: int = 1_000) -> float:
        max_reward = 0

        for _ in tqdm(range(episodes)):
            episode_total_reward = 0

            state, _ = self.__env.reset()
            state = Image.fromarray(state)

            for _ in range(max_steps):
                action, _ = self.__get_action_from_epsilon_greedy(state)
                state, reward, done, _, _ = self.__env.step(action)
                state = Image.fromarray(state)

                episode_total_reward += reward

                if done:
                    break

            max_reward = max(max_reward, episode_total_reward)

        self.__env.reset()
        return max_reward

    def __get_action_from_epsilon_greedy(self, state):
        # Exploration
        if np.random.random() < self.__epsilon:
            return np.random.random_integers(0, self.__n_actions - 1, 1)[
                0
            ], "exploration"

        # Exploitation
        transformed_state = self.__state_transformer(state)
        transformed_state = transformed_state.unsqueeze(0)
        transformed_state = transformed_state.to(self.__device)

        self.__action_value.eval()
        with torch.no_grad():
            return self.__action_value(
                transformed_state
            ).argmax().item(), "exploitation"

    def __save(self):
        print("Saving policy checkpoint...")
        os.makedirs(self.__path, exist_ok=True)

        torch.save(self.__action_value.state_dict(), f"{self.__path}/model.pth")
        dql_parameters = DQLParameters(
            learning_rate=self.__optimizer.param_groups[0]["lr"],
            epsilon=self.__epsilon,
            epsilon_decay=self.__epsilon_decay,
            discount_factor=self.__discount_factor,
            target_update_frequency=self.__target_update_frequency,
            total_episodes=self.__total_episodes,
        )

        dql_parameters.save(f"{self.__path}/policy_parameters.json")
        self.__replay_buffer.save(self.__path)

    def __load(self):
        model_path = f"{self.__path}/model.pth"
        if os.path.exists(model_path):
            self.__action_value.load_state_dict(
                torch.load(
                    model_path,
                    weights_only=True,
                )
            )
        else:
            print("WARNING: Model not found. Using random network weights.")

        policy_parameters_path = f"{self.__path}/policy_parameters.json"

        if os.path.exists(policy_parameters_path):
            dql_parameters = DQLParameters.load(policy_parameters_path)

            self.__optimizer.param_groups[0]["lr"] = dql_parameters.learning_rate
            self.__epsilon = dql_parameters.epsilon
            self.__epsilon_decay = dql_parameters.epsilon_decay
            self.__discount_factor = dql_parameters.discount_factor
            self.__target_update_frequency = dql_parameters.target_update_frequency
            self.__total_episodes = dql_parameters.total_episodes
            np.random.seed(dql_parameters.seed)
        else:
            print("WARNING: Policy parameters not found. Using default values.")

        if os.path.exists(f"{self.__path}/replay_buffer"):
            self.__replay_buffer.load(self.__path)
        else:
            print("WARNING: Replay buffer not found. Using an empty replay buffer")

    def __log(self, episode, step, action, action_type, reward, loss):
        if not os.path.exists(self.__metrics_file):
            os.makedirs(self.__path, exist_ok=True)
            with open(self.__metrics_file, "w") as f:
                f.write(
                    "episode,step,action,action_type,reward,loss,replay_buffer_size\n"
                )

        with open(self.__metrics_file, "a") as f:
            f.write(
                f"{episode},{step},{action},{action_type},{reward},{loss},{len(self.__replay_buffer)}\n"
            )


class DQLParameters:
    def __init__(
        self,
        learning_rate: float = 0.001,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        discount_factor: float = 0.99,
        target_update_frequency: int = 10,
        total_episodes: int = 0,
        seed: int = 0,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.target_update_frequency = target_update_frequency
        self.total_episodes = total_episodes
        self.seed = seed

    def to_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "discount_factor": self.discount_factor,
            "target_update_frequency": self.target_update_frequency,
            "total_episodes": self.total_episodes,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(d):
        return DQLParameters(
            learning_rate=d["learning_rate"],
            epsilon=d["epsilon"],
            epsilon_decay=d["epsilon_decay"],
            discount_factor=d["discount_factor"],
            target_update_frequency=d["target_update_frequency"],
            total_episodes=d["total_episodes"],
            seed=d["seed"],
        )

    def save(self, path: str):
        json.dump(self.to_dict(), open(path, "w"))

    @staticmethod
    def load(path: str):
        return DQLParameters.from_dict(json.load(open(path)))
