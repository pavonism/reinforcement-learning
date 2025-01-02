import gymnasium
import wandb
import ale_py
import torch
import os
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
recordings_file_path = f"{LOG_PATH}/recordings/{current_time}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gymnasium.make("ALE/MsPacman-v5", render_mode="rgb_array")
env = gymnasium.wrappers.RecordVideo(
    env,
    recordings_file_path,
    episode_trigger=lambda episode_id: episode_id % 50 == 0 
)
env = AtariWrapper(env, frame_stack=4, screen_size=84)

input_dim = (4, 84, 84)
action_dim = env.action_space.n
buffer = RolloutBuffer()
max_timesteps = int(2e6)
ppo_agent = PPO(ActorCritic, input_dim, action_dim, buffer, device, total_timesteps=max_timesteps)


wandb.init(
    project="ppo-atari",
    config={
        "env_name": "ALE/MsPacman-v5",
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "clip_epsilon": 0.1,
        "value_coeff": 0.75,
        "entropy_coeff": 0.05,
        "num_epochs": 15,
        "batch_size": 64,
    }
)

episode_rewards = []
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

            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                
                reward_series = pd.Series(episode_rewards)
                rolling_mean = reward_series.rolling(rolling_window_size).mean().iloc[-1] if len(reward_series) > rolling_window_size else None
                expanding_mean = reward_series.expanding().mean().iloc[-1]

                wandb.log({
                    "Episode Reward": episode_reward,
                    "Rolling Mean Reward": rolling_mean,
                    "Expanding Mean Reward": expanding_mean,
                })

                print(f"Episode {episode} | Reward: {episode_reward:.2f} | Timesteps: {time_step}")

                state, _ = env.reset()
                episode_reward = 0

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