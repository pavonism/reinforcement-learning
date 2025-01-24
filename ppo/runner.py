import os
from pathlib import Path
import re
from typing import Optional, TextIO
import pandas as pd
import torch

import gymnasium as gym
from tqdm import tqdm
import wandb

from ppo.algorithm import PPO
from ppo.buffer import RolloutBuffer
from ppo.context import PPOContext


class Runner:
    def __init__(
        self,
        env: gym.Env,
        context: PPOContext,
        ppo: PPO,
        buffer: RolloutBuffer,
        log_file_path: str,
        checkpoint_path: str,
        recordings_path: str,
        checkpoint_interval: int = 500,
    ):
        """
        PPO algorithm runner.

        Args:
            env (gym.Env): Gym environment.
            context (PPOContext): PPO context object containing hyperparameters.
            ppo (PPO): PPO algorithm object.
            buffer (RolloutBuffer): Rollout buffer.
            log_file_path (str): Path to the log file.
            checkpoint_path (str): Path to save the checkpoints.
            recordings_path (str): Path to save the recordings.
            checkpoint_interval (int): Interval to save the checkpoints.
        """

        self.env = env
        self.context = context
        self.ppo = ppo
        self.buffer = buffer
        self.log_file_path = log_file_path
        self.checkpoint_path = checkpoint_path
        self.recordings_path = recordings_path
        self.checkpoint_interval = checkpoint_interval

        self._log_file: Optional[TextIO] = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode = 0
        self._time_step = 0
        self._high_score = float("-inf")

        if os.path.exists(self.log_file_path):
            self._load_log_file()

        if os.path.exists(self.checkpoint_path):
            self._load_checkpoint()

    def run(self, max_timesteps: int, rolling_window_size: int = 100):
        """
        Run the PPO algorithm.

        Args:
            max_timesteps (int): Maximum number of timesteps to run the algorithm.
            rolling_window_size (int): Size of the rolling window to calculate the mean reward.
        """

        self._log_file = open(self.log_file_path, "w", encoding="utf-8")
        p_bar = tqdm(total=self.context.update_interval)

        tqdm.write("Training started...")

        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(self.context.device)

        episode_reward = 0
        curr_episode_length = 0

        while self._time_step < max_timesteps:
            for _ in range(self.context.update_interval):
                action = self.ppo.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                state = next_state
                episode_reward += reward
                self._time_step += 1
                curr_episode_length += 1
                p_bar.update(1)

                if done:
                    if episode_reward > self._high_score:
                        self._high_score = episode_reward
                        self._log_high_score()

                    self._log(
                        rolling_window_size,
                        episode_reward,
                        curr_episode_length,
                    )

                    if self._episode % self.checkpoint_interval == 0:
                        self._save_checkpoint()

                    state, _ = self.env.reset()
                    self._episode += 1
                    episode_reward = 0
                    curr_episode_length = 0
                    p_bar.reset()

            tqdm.write(f"\nUpdating PPO at timestep {self._time_step}...")
            (
                policy_loss,
                value_loss,
                entropy_loss,
                kl_div,
                current_lr,
                current_entropy,
            ) = self.ppo.update(self._time_step)

            wandb.log(
                {
                    "Policy Loss": policy_loss,
                    "Value Loss": value_loss,
                    "Entropy Loss": entropy_loss,
                    "KL Divergence": kl_div,
                    "Learning Rate": current_lr,
                    "Entropy Coefficient": current_entropy,
                    "Total timesteps": self._time_step,
                }
            )

        self.env.close()
        wandb.finish()
        self._log_file.close()
        p_bar.close()
        tqdm.write("Training complete!")

    def _save_checkpoint(self):
        checkpoint = {
            "actor_state_dict": self.ppo.policy.actor.state_dict(),
            "critic_state_dict": self.ppo.policy.critic.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "time_step": self._time_step,
            "buffer": self.ppo.buffer,
            "episode": self._episode,
        }

        checkpoint_file = f"{self.checkpoint_path}/ppo_checkpoint-{self._episode}.pth"
        torch.save(checkpoint, checkpoint_file)
        model_artifact = wandb.Artifact(
            f"checkpoint-episode-{self._episode}", type="model"
        )
        model_artifact.add_file(checkpoint_file)
        wandb.log_artifact(model_artifact)

        videos_artifact = wandb.Artifact(
            f"videos-episode-{self._episode}", type="dataset"
        )
        for file in Path(self.recordings_path).iterdir():
            if file.is_file():
                videos_artifact.add_file(file)
                Path.unlink(file, missing_ok=True)
        wandb.log_artifact(videos_artifact)

        self._log_file.flush()
        os.fsync(self._log_file.fileno())
        log_artifact = wandb.Artifact(f"logs-episode-{self._episode}", type="log")
        log_artifact.add_file(self.log_file_path)
        wandb.log_artifact(log_artifact)

    def _log_high_score(self):
        self._log_file.write(
            f"New High Score! Episode: {self._episode}, High Score: {self._high_score:.2f}\n"
        )
        tqdm.write(f"New High Score: {self._high_score:.2f} in Episode {self._episode}")

    def _log(
        self,
        rolling_window_size: int,
        episode_reward: float,
        curr_episode_length: int,
    ):
        self._episode_rewards.append(episode_reward)
        self._episode_lengths.append(curr_episode_length)

        reward_series = pd.Series(self._episode_rewards)
        length_series = pd.Series(self._episode_lengths)

        rolling_mean = (
            reward_series.rolling(rolling_window_size).mean().iloc[-1]
            if len(reward_series) > rolling_window_size
            else None
        )

        expanding_mean = reward_series.expanding().mean().iloc[-1]

        rolling_mean_length = (
            length_series.rolling(rolling_window_size).mean().iloc[-1]
            if len(length_series) > rolling_window_size
            else None
        )

        expanding_mean_length = length_series.expanding().mean().iloc[-1]

        wandb.log(
            {
                "Episode Reward": episode_reward,
                "Rolling Mean Reward": rolling_mean,
                "Expanding Mean Reward": expanding_mean,
                "Episode Length": curr_episode_length,
                "Rolling Mean Length": rolling_mean_length,
                "Expanding Mean Length": expanding_mean_length,
            }
        )

        self._log_file.write(
            f"Episode {self._episode} | Reward: {episode_reward:.2f} | Timesteps: {curr_episode_length} | Total timesteps: {self._time_step}\n"
        )
        tqdm.write(
            f"Episode {self._episode} | Reward: {episode_reward:.2f} | Timesteps: {curr_episode_length}"
        )

    def _load_log_file(self):
        with open(self.log_file_path, "r") as log_file:
            for line in log_file:
                if "High Score" in line:
                    match = re.search(r"High Score: ([0-9.]+)", line)
                    if match:
                        self._high_score = max(self._high_score, float(match.group(1)))
                    continue
                match = re.search(
                    r"Episode (\d+) \| Reward: ([0-9.]+) \| Timesteps: (\d+)", line
                )
                if match:
                    reward = float(match.group(2))
                    timesteps = int(match.group(3))
                    self._episode_rewards.append(reward)
                    self._episode_lengths.append(timesteps)

    def _load_checkpoint(self):
        checkpoint_files = sorted(
            Path(self.checkpoint_path).glob("ppo_checkpoint-*.pth"),
            key=os.path.getmtime,
        )

        if not checkpoint_files:
            tqdm.write("No checkpoint files found in the checkpoint path.")
            return

        latest_checkpoint_file = checkpoint_files[-1]
        checkpoint_file = latest_checkpoint_file

        match = re.search(r"ppo_checkpoint-(\d+)", str(checkpoint_file))
        if match:
            loaded_episodes = int(match.group(1))
            self._episode = loaded_episodes

        checkpoint = torch.load(
            checkpoint_file,
            map_location=self.context.device,
            weights_only=False,
        )

        self.ppo.policy.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.ppo.policy.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self._time_step = checkpoint["time_step"] if "time_step" in checkpoint else 0
        self.buffer = checkpoint.get("buffer", self.buffer)
        self.buffer.clear()
        self.ppo.buffer = self.buffer

        wandb.config.update({"starting_episode": self._episode}, allow_val_change=True)
        print(f"Checkpoint loaded: {checkpoint_file}")
