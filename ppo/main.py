import gymnasium
import ale_py
import pandas as pd
import torch
import os
from ppo.actor_critic import ActorCritic
from ppo.buffer import RolloutBuffer
from ppo.ppo_agent import PPO
from ppo.atari_wrapper import AtariWrapper
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create directories
LOG_PATH = "checkpoints/ppo/5mln-larger-smaller-entropy"
os.makedirs(f"{LOG_PATH}/recordings", exist_ok=True)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = f"{LOG_PATH}/training_log_{current_time}.txt"
reward_file_path = f"{LOG_PATH}/rewards_{current_time}.csv"
loss_file_path = f"{LOG_PATH}/loss_{current_time}.csv"
recordings_file_path = f"{LOG_PATH}/recordings/{current_time}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment with proper wrapping order
env = gymnasium.make("ALE/MsPacman-v5", render_mode="rgb_array")
env = gymnasium.wrappers.RecordVideo(
    env, recordings_file_path, episode_trigger=lambda episode_id: True
)
env = AtariWrapper(env, frame_stack=4, screen_size=84)

input_dim = (4, 84, 84)
action_dim = env.action_space.n
buffer = RolloutBuffer()
ppo_agent = PPO(ActorCritic, input_dim, action_dim, buffer, device, loss_file_path)

# Training loop
with open(log_file_path, "w", encoding="utf-8") as log_file:
    max_timesteps = int(5e6)
    update_timestep = 4096
    time_step = 0
    episode_rewards = []
    episode = 0

    print("Starting training...")
    log_file.write(f"episode,step,reward\n")
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

            log_file.write(f"{episode},{time_step},{reward}\n")

            if done:
                episode += 1
                log_file.flush()

                print(
                    f"Episode {episode} | Reward: {episode_reward:.2f} | Timesteps: {time_step}"
                )

                if episode % 100 == 0:
                    print(f"Saving model at episode {episode}...")
                    ppo_agent.save(f"{LOG_PATH}/ppo_model.pt")

                state, _ = env.reset()
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                episode_reward = 0

        print(f"\nUpdating PPO at timestep {time_step}...")
        ppo_agent.update()

    env.close()
    print("Training complete!")

ppo_agent.save(f"{LOG_PATH}/ppo_model.pt")
