import os
import time
import ale_py
import gymnasium

from ppo.actor_critic import ActorCritic
from ppo.buffer import RolloutBuffer
from ppo.algorithm import PPO
from ppo.context import PPOContext
from ppo.runner import Runner
from ppo.utils import init_wandb, log_networks_weights_and_biases, prepare_atari_env


LOG_PATH = os.path.abspath("./checkpoints/ppo")

CHECKPOINT_TIMESTAMP = int(time.time())
RECORDINGS_PATH = f"{LOG_PATH}/recordings/{CHECKPOINT_TIMESTAMP}"
LOG_FILE_PATH = f"{LOG_PATH}/training_log.txt"
CHECKPOINT_PATH = f"{LOG_PATH}/checkpoints/"

os.makedirs(f"{LOG_PATH}/recordings", exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

gymnasium.register_envs(ale_py)
env = prepare_atari_env(RECORDINGS_PATH)

context = PPOContext()
buffer = RolloutBuffer()
actor_critic = ActorCritic(input_dim=(4, 84, 84), action_dim=env.action_space.n)
ppo = PPO(context, actor_critic, buffer)

log_networks_weights_and_biases(ppo.policy.actor, ppo.policy.critic)
init_wandb()

runner = Runner(
    env=env,
    context=context,
    ppo=ppo,
    buffer=buffer,
    log_file_path=LOG_FILE_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    recordings_path=RECORDINGS_PATH,
)

runner.run(max_timesteps=2e7, rolling_window_size=20)
