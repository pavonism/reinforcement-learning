import gymnasium
import ale_py
import torch
import os
from ppo.actor_critic import ActorCritic
from ppo.buffer import RolloutBuffer
from ppo.ppo_agent import PPO
from ppo.atari_wrapper import AtariWrapper
from datetime import datetime

# Create directories
LOG_PATH = "checkpoints/ppo"
os.makedirs(f"{LOG_PATH}/recordings", exist_ok=True)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = f"{LOG_PATH}/training_log_{current_time}.txt"
recordings_file_path = f"{LOG_PATH}/recordings/{current_time}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment with proper wrapping order
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
ppo_agent = PPO(ActorCritic, input_dim, action_dim, buffer, device)

# Training loop
with open(log_file_path, "w", encoding="utf-8") as log_file:
    max_timesteps = int(1e6)
    update_timestep = 2048
    time_step = 0
    episode_rewards = []
    episode = 0
    
    print("Starting training...")
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
                log_file.write(f"{episode},{episode_reward}\n")
                log_file.flush()
                
                print(f"Episode {episode} | Reward: {episode_reward:.2f} | Timesteps: {time_step}")
                
                state, _ = env.reset()
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                episode_reward = 0
        
        print(f"\nUpdating PPO at timestep {time_step}...")
        ppo_agent.update()

    env.close()
    print("Training complete!")