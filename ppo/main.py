import gymnasium
import wandb
import ale_py
import torch
import os
from pathlib import Path
from ppo.actor_critic import ActorCritic
from ppo.buffer import RolloutBuffer
from ppo.ppo_agent import PPO
from ppo.atari_wrapper import AtariWrapper
from datetime import datetime
import pandas as pd

LOG_PATH = os.path.abspath("./checkpoints/ppo")
os.makedirs(f"{LOG_PATH}/recordings", exist_ok=True)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = f"{LOG_PATH}/training_log_{current_time}.txt"
checkpoint_path = f"{LOG_PATH}/checkpoints/{current_time}"
recordings_path = f"{LOG_PATH}/recordings/{current_time}"
os.makedirs(checkpoint_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gymnasium.make("ALE/MsPacman-v5", render_mode="rgb_array")
env = gymnasium.wrappers.RecordVideo(
    env,
    recordings_path,
    episode_trigger=lambda episode_id: True
)
env = AtariWrapper(env, frame_stack=4, screen_size=84)

high_score = float('-inf')
checkpoint_interval = 10

input_dim = (4, 84, 84)
action_dim = env.action_space.n
buffer = RolloutBuffer()
max_timesteps = int(1e7)
ppo_agent = PPO(ActorCritic, input_dim, action_dim, buffer, device, total_timesteps=max_timesteps)

wandb.init(
    project="ppo-atari",
    config={
        "env_name": "ALE/MsPacman-v5",
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "clip_epsilon": 0.1,
        "value_coeff": 1,
        "entropy_coeff": 0.05,
        "num_epochs": 15,
        "batch_size": 256,
    }
)

episode_rewards, episode_lengths = [], []
rolling_window_size = 20

# Training loop
with open(log_file_path, "w", encoding="utf-8") as log_file:
    update_timestep = 4096
    time_step = 0
    episode_rewards = []
    episode = 0
    
    print("Starting training! ")
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)
    
    episode_reward = 0
    curr_episode_length = 0
    
    while time_step < max_timesteps:
        for _ in range(update_timestep):
            action = ppo_agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)

            state = next_state
            episode_reward += reward
            time_step += 1
            curr_episode_length += 1

            if done:
                if episode_reward > high_score:
                    high_score = episode_reward
                    log_file.write(f"New High Score! Episode: {episode}, High Score: {high_score:.2f}\n")
                    print(f"New High Score: {high_score:.2f} in Episode {episode}")
                episode_rewards.append(episode_reward)
                episode_lengths.append(curr_episode_length)
                
                reward_series = pd.Series(episode_rewards)
                length_series = pd.Series(episode_lengths)
                
                rolling_mean = reward_series.rolling(rolling_window_size).mean().iloc[-1] if len(reward_series) > rolling_window_size else None
                expanding_mean = reward_series.expanding().mean().iloc[-1]
                rolling_mean_length = length_series.rolling(rolling_window_size).mean().iloc[-1] if len(length_series) > rolling_window_size else None
                expanding_mean_length = length_series.expanding().mean().iloc[-1]

                wandb.log({
                    "Episode Reward": episode_reward,
                    "Rolling Mean Reward": rolling_mean,
                    "Expanding Mean Reward": expanding_mean,
                    "Episode Length": curr_episode_length,
                    "Rolling Mean Length": rolling_mean_length,
                    "Expanding Mean Length": expanding_mean_length
                })

                log_file.write(f"Episode {episode} | Reward: {episode_reward:.2f} | Timesteps: {curr_episode_length} | Total timesteps: {time_step}\n")
                print(f"Episode {episode} | Reward: {episode_reward:.2f} | Timesteps: {curr_episode_length}")
                
                if(episode % checkpoint_interval == 0):
                    checkpoint = {
                        "actor_state_dict": ppo_agent.policy.actor.state_dict(),
                        "critic_state_dict": ppo_agent.policy.critic.state_dict(),
                        "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
                        "time_step": time_step,  # Save current timestep
                        "buffer": ppo_agent.buffer  # Optionally save the buffer if continuing training without resetting it
                    }

                    checkpoint_file = f"{checkpoint_path}/ppo_checkpoint-{episode}.pth"
                    torch.save(checkpoint, checkpoint_file)
                    model_artifact = wandb.Artifact(f"checkpoint-episode-{episode}", type="model")
                    model_artifact.add_file(checkpoint_file)
                    wandb.log_artifact(model_artifact)
                    
                    videos_artifact = wandb.Artifact(f"videos-episode-{episode}", type="dataset")
                    for file in Path(recordings_path).iterdir():
                        if file.is_file():
                            videos_artifact.add_file(file)
                    wandb.log_artifact(videos_artifact)
                    
                    log_file.flush()
                    os.fsync(log_file.fileno())
                    log_artifact = wandb.Artifact(f"logs-episode-{episode}", type="log")
                    log_artifact.add_file(log_file_path)
                    wandb.log_artifact(log_artifact)

                state, _ = env.reset()
                episode += 1
                episode_reward = 0
                curr_episode_length = 0

        print(f"\nUpdating PPO at timestep {time_step}...")
        policy_loss, value_loss, entropy_loss, kl_div, current_lr, current_entropy = ppo_agent.update(time_step)


        wandb.log({
            "Policy Loss": policy_loss,
            "Value Loss": value_loss,
            "Entropy Loss": entropy_loss,
            "KL Divergence": kl_div,
            "Learning Rate": current_lr,
            "Entropy Coefficient": current_entropy,
            "Total timesteps": time_step,
        })

    env.close()
    wandb.finish()
    print("Training complete!")