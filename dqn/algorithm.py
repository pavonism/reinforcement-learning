import copy
import json
import os
from typing import Tuple

from gymnasium import Env
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from dqn.context import Context
from dqn.replay import ReplayBuffer, TensorBatchExperience


class DQNAlgorithm:
    def __init__(
        self,
        context: Context,
        env: Env,
        network: nn.Module,
        path: str,
        ddqn: bool = False,
        seed: int = 0,
    ):
        """
        Deep Q-Learning algorithm implementation.

        Args:
            context (Context): Context object containing hyperparameters.
            env (Env): Gym environment.
            network (nn.Module): Neural network to approximate the Q-values.
            path (str): Path to save / load the policy checkpoint.
            ddqn (bool): Whether to use Double DQN.
        """
        self._context = context
        self._env = env
        self._action_value = network.to(context.device)
        self._target_action_value = copy.deepcopy(network)
        self._path = path
        self._ddqn = ddqn

        self._optimizer = torch.optim.Adam(
            self._action_value.parameters(),
            lr=context.learning_rate,
        )

        self._replay_buffer = ReplayBuffer(
            seed=seed,
            capacity=context.replay_buffer_size,
            with_priorities=context.priority_replay,
        )

        self._total_episodes = 0
        self._epsilon = context.epsilon
        np.random.seed(seed)

        if os.path.exists(self._path):
            self._load()

    def play(self, episodes: int = 1, max_steps: int = 1_000) -> float:
        """
        Play the environment using the trained policy.

        Args:
            episodes (int): Number of episodes to play the environment.
            max_steps (int): Maximum number of steps per episode.
        """

        max_reward = 0

        for _ in tqdm(range(episodes)):
            episode_total_reward = 0

            state, _ = self._env.reset()
            state = torch.from_numpy(state)

            for _ in range(max_steps):
                action, _ = self._use_epsilon_greedy(state)
                new_state, reward, done, _, _ = self._env.step(action)
                state = torch.from_numpy(new_state)

                episode_total_reward += reward

                if done:
                    break

            max_reward = max(max_reward, episode_total_reward)

        self._env.reset()
        return max_reward

    def train(self, episodes: int = 1, max_steps: int = 2_000):
        """
        Train the DQN policy.

        Args:
            episodes (int): Number of episodes to train the policy.
            max_steps (int): Maximum number of steps per episode.
        """
        target_update_counter = 0

        with tqdm(total=max_steps) as pbar:
            for episode in range(self._total_episodes, self._total_episodes + episodes):
                reward, steps, loss = self._train_episode(max_steps, pbar)

                target_update_counter += steps
                self._total_episodes += 1

                if self._epsilon > self._context.min_epsilon:
                    self._epsilon *= self._context.epsilon_decay

                if episode % self._context.save_frequency_in_episodes == 0:
                    self._save()

                if target_update_counter > self._context.target_update_frequency:
                    target_update_counter = 0
                    self._update_target_action_value_network()

                wandb.log(
                    {
                        "episode": episode,
                        "epsilon": self._epsilon,
                        "episode_length": steps,
                        "episode_reward": reward,
                        "episode_loss": loss,
                    }
                )

            self._env.reset()
            self._save()

    def _train_episode(self, max_steps: int, pbar: tqdm) -> Tuple[float, int, float]:
        episode_total_reward = 0
        episode_steps = 0
        all_losses = []

        pbar.reset()
        state, *_ = self._env.reset()
        state = torch.from_numpy(state)

        for _ in range(max_steps):
            action = self._use_epsilon_greedy(state)
            new_state, reward, done, *_ = self._env.step(action)
            new_state = torch.from_numpy(new_state)

            self._replay_buffer.save(state, action, reward, new_state, done)
            state = new_state

            experiences = self._replay_buffer.sample(
                self._context.device,
                stack_frame=self._context.stack_frame,
                batch_size=self._context.batch_size,
            )

            loss = self._train_step(experiences)
            all_losses.append(loss)

            episode_total_reward += reward
            episode_steps += 1

            wandb.log({"loss": loss})
            pbar.update(1)

            if done:
                break

        episode_loss = np.mean(all_losses)

        return episode_total_reward, episode_steps, episode_loss

    def _train_step(self, experiences: TensorBatchExperience) -> float:
        q_values = (
            self._action_value(experiences.states)
            .gather(1, experiences.actions.view(-1, 1))
            .flatten()
        )

        if self._ddqn:
            next_action_indexes = self._action_value(
                experiences.next_states,
            ).argmax(1)

            target_q_values = (
                self._target_action_value(experiences.next_states)
                .gather(1, next_action_indexes.view(-1, 1))
                .flatten()
            )
        else:
            target_q_values = self._target_action_value(
                experiences.next_states,
            ).max(1)[0]

        y = (
            experiences.rewards
            + (1 - experiences.dones)
            * self._context.discount_factor
            * target_q_values.detach()
        )

        td_error = (y - q_values).abs().detach().type(torch.double)
        self._replay_buffer.update_priorities(experiences.indexes, td_error)

        self._action_value.train()
        self._optimizer.zero_grad()
        loss = F.mse_loss(y, q_values)
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _update_target_action_value_network(self):
        tqdm.write("Updating target network...")
        self._target_action_value.load_state_dict(self._action_value.state_dict())

    def _use_epsilon_greedy(self, state: torch.Tensor) -> int:
        # Exploration
        if np.random.random() < self._epsilon:
            return np.random.random_integers(0, self._context.n_actions - 1, 1)[0]

        # Exploitation
        last_states = self._replay_buffer.get_last_states(self._context.stack_frame - 1)

        states = torch.concat([last_states, state.unsqueeze(0)])
        states = states.unsqueeze(0).to(self._context.device)

        self._action_value.eval()
        with torch.no_grad():
            q_values = self._action_value(states).squeeze(0)
            return q_values.argmax().item()

    def _save(self):
        os.makedirs(self._path, exist_ok=True)
        self._replay_buffer.save_to_directory(self._path)

        with open(f"{self._path}/metrics.csv", "a") as f:
            json.dump(
                {
                    "total_episodes": self._total_episodes,
                    "epsilon": self._epsilon,
                    "replay_buffer_position": self._replay_buffer.position,
                },
                f,
            )

        model_path = f"{self._path}/model.pth"
        torch.save(self._action_value.state_dict(), model_path)
        tqdm.write("Checkpoint saved.")

    def _load(self):
        tqdm.write("Loading checkpoint...")
        self._load_weights()
        self._load_policy_parameters()
        self._load_replay_buffer()

    def _load_weights(self):
        model_path = f"{self._path}/model.pth"
        if os.path.exists(model_path):
            self._action_value.load_state_dict(
                torch.load(
                    model_path,
                    weights_only=True,
                )
            )
        else:
            tqdm.write("WARNING: Model not found. Using random network weights.")

    def _load_policy_parameters(self):
        policy_parameters_path = f"{self._path}/policy_parameters.json"

        if os.path.exists(policy_parameters_path):
            with open(policy_parameters_path, "r") as f:
                policy_parameters = json.load(f)

            self._epsilon = policy_parameters.get("epsilon", self._epsilon)
            self._total_episodes = policy_parameters.get(
                "total_episodes", self._total_episodes
            )
            self._replay_buffer.position = policy_parameters.get(
                "replay_buffer_position", self._replay_buffer.position
            )
        else:
            tqdm.write("WARNING: Policy parameters not found. Using default values.")

    def _load_replay_buffer(self):
        replay_buffer_path = f"{self._path}/replay_buffer.pt"
        if os.path.exists(replay_buffer_path):
            self._replay_buffer.load(self._path)
        else:
            tqdm.write("WARNING: Replay buffer not found. Using an empty replay buffer")
