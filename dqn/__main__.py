import time
import ale_py
import gymnasium
import torch
import wandb

from dqn.algorithm import DQNAlgorithm
from dqn.context import Context
from dqn.networks import DQN

CHECKPOINT_PATH = "checkpoints/dqn"
CHECKPOINT_TIMESTAMP = int(time.time())

gymnasium.register_envs(ale_py)
wandb.login()
wandb.init(project="muzero")
torch.set_printoptions(profile="full")


env = gymnasium.wrappers.AtariPreprocessing(
    gymnasium.wrappers.RecordVideo(
        gymnasium.make("ALE/MsPacman-v5", render_mode="rgb_array", frameskip=4),
        f"{CHECKPOINT_PATH}/recordings_{CHECKPOINT_TIMESTAMP}",
        lambda x: True,
    ),
    screen_size=96,
    grayscale_obs=True,
    frame_skip=1,
    scale_obs=True,
)
n_actions = env.action_space.n

context = Context(n_actions=n_actions, priority_replay=True)
dqn = DQN(n_actions=n_actions, stack_frame=context.stack_frame)

algorithm = DQNAlgorithm(
    env=env,
    context=context,
    network=dqn,
    path=CHECKPOINT_PATH,
    ddqn=True,
)

algorithm.train(episodes=1_000_000)
