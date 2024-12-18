import gymnasium
import ale_py
import torch
from ppo.actor_critic import ActorCritic
from ppo.buffer import RolloutBuffer
from ppo.ppo_agent import PPO
from ppo.atari_wrapper import AtariWrapper

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = AtariWrapper(gymnasium.make("ALE/MsPacman-v5"))

input_dim = (4, 84, 84)
action_dim = env.action_space.n
buffer = RolloutBuffer()

ppo_agent = PPO(ActorCritic, input_dim, action_dim, buffer, device)

# Training Loop
max_timesteps = 1e6
update_timestep = 2048
time_step = 0
episode_rewards = []
episode = 0
print("Starting...")
while time_step < max_timesteps:
    state = env.reset()
    episode_reward = 0

    for _ in range(update_timestep):
        action = ppo_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        buffer.rewards.append(reward)
        buffer.is_terminals.append(done)

        state = next_state
        episode_reward += reward
        time_step += 1

        if done:
            episode += 1
            episode_rewards.append(episode_reward)
            print(f"Episode {episode} | Reward: {episode_reward} | Total Timesteps: {time_step}")
            state = env.reset()
            episode_reward = 0

    # PPO Update
    print(f"\nUpdating PPO... Time Step: {time_step}")
    ppo_agent.update()

# Environment cleanup
env.close()

print("Training complete!")

