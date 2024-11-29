import gymnasium as gym
import ale_py
from pandas import Timestamp

from src.policies.dql_policy import DQLPolicy
from src.networks.dqn import DQN
from src.wrappers import TerminateOnLifeLossWrapper

ATTEMPT_NAME = "ddqn/terminate_on_life_loss"

output_path = f"../outputs/{ATTEMPT_NAME}/{int(Timestamp.now().timestamp())}"

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, output_path, episode_trigger=lambda _: True)
env = TerminateOnLifeLossWrapper(env)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n
print(f"Observation shape: {obs_shape}")
print(f"Number of actions: {n_actions}")

policy = DQLPolicy(
    env,
    model=DQN(n_actions),
    path=f"../checkpoints/{ATTEMPT_NAME}",
    min_epsilon=0.001,
    ddqn=True,
)

total_reward = policy.train(episodes=10_000, max_steps=10_000)
