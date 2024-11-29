import copy
import json
import os
from gymnasium import Env
import pandas as pd
from torch import nn
import torch
from tqdm import tqdm
from src.policies.common import Policy, ReplayBuffer
import numpy as np
from torchvision import transforms
from PIL import Image

from line_profiler import profile


class DQLPolicy(Policy):
    def __init__(
        self,
        env: Env,
        model: nn.Module,
        path: str,
        state_transformer: transforms.Compose = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.squeeze(0)),
            ]
        ),
        learning_rate: float = 0.0001,
        epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        discount_factor: float = 0.99,
        target_update_frequency: int = 10_000,
        save_frequency_in_episodes: int = 100,
        batch_size: int = 128,
        experience_play_batch_size=8,
        priority_replay_buffer: bool = False,
        replay_buffer_capacity: int = 800_000,
        ddqn: bool = False,
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
        self.__min_epsilon = min_epsilon
        self.__epsilon_decay = epsilon_decay
        self.__discount_factor = discount_factor
        self.__target_update_frequency = target_update_frequency
        self.__save_frequency_in_episodes = save_frequency_in_episodes
        self.__batch_size = batch_size
        self.__experience_play_batch_size = experience_play_batch_size
        self.__replay_buffer = ReplayBuffer(
            seed=seed,
            capacity=replay_buffer_capacity,
            with_priorities=priority_replay_buffer,
        )
        self.__total_episodes = 0
        self.__ddqn = ddqn
        np.random.seed(seed)

        if os.path.exists(self.__path):
            self.__load()

    @profile
    def train(self, episodes: int = 1, max_steps: int = 1_000) -> float:
        max_reward = 0

        for episode in range(self.__total_episodes, self.__total_episodes + episodes):
            state, *_ = self.__env.reset()
            state = self.__state_transformer(Image.fromarray(state))
            episode_total_reward = 0
            target_update_counter = 0

            for step in range(max_steps):
                action, action_type, epsilon_greedy_q_values = (
                    self.__get_action_from_epsilon_greedy(state)
                )
                new_state, reward, done, *_ = self.__env.step(action)
                new_state = self.__state_transformer(Image.fromarray(new_state))

                self.__replay_buffer.remember_experience(
                    state, action, reward, new_state, done
                )
                state = new_state

                experiences = self.__replay_buffer.sample_experience(
                    self.__device,
                    state_memory_batch_size=self.__experience_play_batch_size,
                    batch_size=self.__batch_size,
                )

                q_values = (
                    self.__action_value(experiences.states)
                    .gather(1, experiences.actions.view(-1, 1))
                    .flatten()
                )

                if self.__ddqn:
                    next_action_indexes = self.__action_value(
                        experiences.next_states,
                    ).argmax(1)

                    target_q_values = (
                        self.__target_action_value(experiences.next_states)
                        .gather(1, next_action_indexes.view(-1, 1))
                        .flatten()
                    )
                else:
                    target_q_values = self.__target_action_value(
                        experiences.next_states,
                    ).max(1)[0]

                y_i = (
                    experiences.rewards
                    + (1 - experiences.dones) * self.__discount_factor * target_q_values
                )

                td_error = (y_i - q_values).abs().detach().type(torch.double)
                self.__replay_buffer.update_priorities(experiences.indexes, td_error)

                self.__action_value.train()
                self.__optimizer.zero_grad()
                loss = nn.MSELoss()
                output = loss(y_i, q_values)
                output.backward()
                self.__optimizer.step()

                episode_total_reward += reward
                target_update_counter += 1

                if target_update_counter % self.__target_update_frequency == 0:
                    target_update_counter = 0
                    self.__update_target_action_value_network()

                self.__log(
                    episode,
                    step,
                    action,
                    action_type,
                    reward,
                    output.item(),
                    epsilon_greedy_q_values,
                )

                if done:
                    break

            print(f"Total reward for episode {episode}: {episode_total_reward}")
            max_reward = max(max_reward, episode_total_reward)

            if self.__epsilon > self.__min_epsilon:
                self.__epsilon *= self.__epsilon_decay
            self.__total_episodes += 1

            if episode % self.__save_frequency_in_episodes == 0:
                self.__save()

        self.__env.reset()
        self.__save()
        return max_reward

    def __update_target_action_value_network(self):
        self.__target_action_value.load_state_dict(self.__action_value.state_dict())

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
            return (
                np.random.random_integers(0, self.__n_actions - 1, 1)[0],
                "exploration",
                [q_value / self.__n_actions for q_value in [1] * self.__n_actions],
            )

        # Exploitation
        last_states = self.__replay_buffer.get_last_states(
            self.__experience_play_batch_size - 1
        )

        states = torch.concat([last_states, state.unsqueeze(0)])
        states = states.unsqueeze(0).to(self.__device)

        self.__action_value.eval()
        with torch.no_grad():
            q_values = self.__action_value(states).squeeze(0)
            return q_values.argmax().item(), "exploitation", q_values.tolist()

    def __save(self):
        print("Saving policy checkpoint...")
        os.makedirs(self.__path, exist_ok=True)
        self.__replay_buffer.save(self.__path)

        dql_parameters = DQLParameters(
            learning_rate=self.__optimizer.param_groups[0]["lr"],
            epsilon=self.__epsilon,
            epsilon_decay=self.__epsilon_decay,
            discount_factor=self.__discount_factor,
            target_update_frequency=self.__target_update_frequency,
            total_episodes=self.__total_episodes,
            batch_size=self.__batch_size,
            ddqn=self.__ddqn,
        )

        dql_parameters.save(f"{self.__path}/policy_parameters.json")
        torch.save(self.__action_value.state_dict(), f"{self.__path}/model.pth")

    def __load(self):
        print("Loading policy checkpoint...")
        self.__load_weights()
        self.__load_policy_parameters()
        self.__load_replay_buffer()
        self.__cut_metrics()
        self.__set_replay_buffer_position()

    def __set_replay_buffer_position(self):
        if os.path.exists(self.__metrics_file):
            metrics_df = pd.read_csv(self.__metrics_file)
            self.__replay_buffer.position = len(metrics_df) % len(
                self.__replay_buffer.priorities
            )
            print(f"Replay buffer position set to {self.__replay_buffer.position}")

    def __load_weights(self):
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

    def __load_policy_parameters(self):
        policy_parameters_path = f"{self.__path}/policy_parameters.json"

        if os.path.exists(policy_parameters_path):
            dql_parameters = DQLParameters.load(policy_parameters_path)

            self.__optimizer.param_groups[0]["lr"] = dql_parameters.learning_rate
            self.__epsilon = dql_parameters.epsilon
            self.__epsilon_decay = dql_parameters.epsilon_decay
            self.__discount_factor = dql_parameters.discount_factor
            self.__target_update_frequency = dql_parameters.target_update_frequency
            self.__total_episodes = dql_parameters.total_episodes
            self.__replay_buffer.with_priorities = dql_parameters.priority_replay_buffer
            self.__batch_size = dql_parameters.batch_size
            self.__ddqn = dql_parameters.ddqn
            np.random.seed(dql_parameters.seed)
        else:
            print("WARNING: Policy parameters not found. Using default values.")

    def __load_replay_buffer(self):
        if os.path.exists(f"{self.__path}/replay_buffer.pt"):
            self.__replay_buffer.load(self.__path)
        else:
            print("WARNING: Replay buffer not found. Using an empty replay buffer")

    def __cut_metrics(self):
        if os.path.exists(self.__metrics_file):
            metrics_df = pd.read_csv(self.__metrics_file)
            episodes_in_metrics = metrics_df.groupby("episode")

            if len(episodes_in_metrics) > self.__total_episodes:
                print(
                    "WARNING: Metrics file contains more episodes than the total episodes in the checkpoint files. Cutting the metrics file...."
                )
                metrics_df = metrics_df[metrics_df["episode"] < self.__total_episodes]
                metrics_df.to_csv(self.__metrics_file, index=False)

    def __log(self, episode, step, action, action_type, reward, loss, q_values):
        if not os.path.exists(self.__metrics_file):
            os.makedirs(self.__path, exist_ok=True)
            with open(self.__metrics_file, "w") as f:
                headers = [
                    "episode",
                    "step",
                    "action",
                    "action_type",
                    "reward",
                    "loss",
                    "replay_buffer_size",
                ] + [f"q_value_{i}" for i in range(self.__n_actions)]
                f.write(",".join(headers) + "\n")

        with open(self.__metrics_file, "a") as f:
            q_values = ",".join(map(str, q_values))
            f.write(
                f"{episode},{step},{action},{action_type},{reward},{loss},{len(self.__replay_buffer)},{q_values}\n"
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
        batch_size: int = 128,
        priority_replay_buffer: bool = False,
        ddqn: bool = False,
        seed: int = 0,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.target_update_frequency = target_update_frequency
        self.total_episodes = total_episodes
        self.batch_size = batch_size
        self.priority_replay_buffer = priority_replay_buffer
        self.ddqn = ddqn
        self.seed = seed

    def to_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "discount_factor": self.discount_factor,
            "target_update_frequency": self.target_update_frequency,
            "total_episodes": self.total_episodes,
            "batch_size": self.batch_size,
            "priority_replay_buffer": self.priority_replay_buffer,
            "ddqn": self.ddqn,
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
            batch_size=d["batch_size"],
            priority_replay_buffer=d.get("priority_replay_buffer", False),
            ddqn=d.get("ddqn", False),
            seed=d["seed"],
        )

    def save(self, path: str):
        json.dump(self.to_dict(), open(path, "w"))

    @staticmethod
    def load(path: str):
        return DQLParameters.from_dict(json.load(open(path)))
