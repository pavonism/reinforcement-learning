import gymnasium as gym
import ale_py
from pandas import Timestamp

from src.policies.dql_policy import DQLPolicy
from src.networks.dqn import DQN

output_path = f"../outputs/DQN/{int(Timestamp.now().timestamp())}"

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, output_path, episode_trigger=lambda _: True)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n
print(f"Observation shape: {obs_shape}")
print(f"Number of actions: {n_actions}")

policy = DQLPolicy(
    env, model=DQN(n_actions), path="../checkpoints/dqn/priority_replay_buffer"
)
total_reward = policy.train(episodes=10_000, max_steps=10_000)
