import multiprocessing
import os

import gymnasium
import ale_py

from muzero.context import MuZeroContext
from muzero.networks import MuZeroNetwork
from muzero.replay import ReplayBuffer
from muzero.threads import Actor, GamesCollector, SharedContext

CHECKPOINT_PATH = "checkpoints/muzero"
games_queue = multiprocessing.Queue()
stop_event = multiprocessing.Event()
replay_buffer = ReplayBuffer(capacity=1500)

gymnasium.register_envs(ale_py)


def env_factory(actor_id: int):
    return gymnasium.wrappers.AtariPreprocessing(
        gymnasium.wrappers.RecordVideo(
            gymnasium.make("ALE/MsPacman-v5", render_mode="rgb_array"),
            f"{CHECKPOINT_PATH}/recordings/{actor_id}",
        ),
        screen_size=96,
        grayscale_obs=False,
        frame_skip=1,
    )


context = MuZeroContext(
    n_actions=9,
    max_moves=27000,  # Half an hour at action repeat 4.
    discount=0.997,
    dirichlet_alpha=0.25,
    num_simulations=50,
    batch_size=1024,
    td_steps=10,
    num_actors=1,
    lr_init=0.05,
    lr_decay_steps=350e3,
    env_factory=env_factory,
    checkpoint_path=CHECKPOINT_PATH,
)

network = (
    MuZeroNetwork.from_checkpoint(CHECKPOINT_PATH)
    if os.path.exists(CHECKPOINT_PATH)
    else MuZeroNetwork(
        raw_state_channels=3,
        hidden_state_channels=128,
        num_actions=context.n_actions,
        value_support_size=601,
        reward_support_size=601,
    )
)

shared_context = SharedContext(
    games_queue=games_queue,
    network=network,
    stop_event=stop_event,
)

experience_collector = GamesCollector(
    queue=games_queue,
    replay_buffer=replay_buffer,
    save_frequency=50,
    path=f"{CHECKPOINT_PATH}/replay_buffer.gzip",
)

# experience_collector.start()

for actor_id in range(context.num_actors):
    actor = Actor(
        actor_id=actor_id,
        context=context,
        shared_context=shared_context,
    )
    actor.start()

actor.join()
