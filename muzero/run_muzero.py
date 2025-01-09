from queue import Queue
import threading
import os
import signal

import gymnasium
import ale_py
import torch
from tqdm import tqdm
import wandb
import time

from muzero.context import MuZeroContext
from muzero.networks import MuZeroNetwork
from muzero.replay import ReplayBuffer
from muzero.threads import Actor, GamesCollector, Reanalyzer, SharedContext, Trainer

CHECKPOINT_PATH = "checkpoints/muzero_fixes_priorities"
CHECKPOINT_TIMESTAMP = int(time.time())
games_queue = Queue()
stop_event = threading.Event()
train_device = "cuda" if torch.cuda.is_available() else "cpu"
actor_device = "cpu"
print("Training on", train_device)
print("Acting on", actor_device)

gymnasium.register_envs(ale_py)
wandb.login()
wandb.init(project="muzero", id="lh45seoh", resume="must")
torch.set_printoptions(profile="full")


def env_factory(actor_id: int):
    return gymnasium.wrappers.AtariPreprocessing(
        gymnasium.wrappers.RecordVideo(
            gymnasium.make("ALE/MsPacman-v5", render_mode="rgb_array", frameskip=4),
            f"{CHECKPOINT_PATH}/recordings_{CHECKPOINT_TIMESTAMP}/{actor_id}",
            lambda x: True,
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
    # num_simulations=50,
    num_simulations=10,
    # batch_size=1024,
    batch_size=256,
    # td_steps=10,
    td_steps=5,
    num_actors=1,
    lr_init=0.05,
    lr_decay_steps=350e3,
    env_factory=env_factory,
    checkpoint_path=CHECKPOINT_PATH,
    train_device=train_device,
    value_loss_weight=0.25,
)

replay_buffer = ReplayBuffer(
    capacity=200,
    td_steps=context.td_steps,
    unroll_steps=context.num_unroll_steps,
    path=f"{CHECKPOINT_PATH}/replay_buffer.gzip",
)

network = (
    MuZeroNetwork.from_checkpoint(CHECKPOINT_PATH)
    if os.path.exists(f"{CHECKPOINT_PATH}/muzero_network.pt")
    else MuZeroNetwork(
        raw_state_channels=context.n_states_representation * 3
        + context.n_actions_representation,
        hidden_state_channels=256,
        num_actions=context.n_actions,
        value_support_size=601,
        reward_support_size=601,
    )
).to(actor_device)

shared_context = SharedContext(
    games_queue=games_queue,
    network=network,
    stop_event=stop_event,
    replay_buffer=replay_buffer,
)

for actor_id in range(context.num_actors):
    actor = Actor(
        actor_id=actor_id,
        context=context,
        shared_context=shared_context,
    )
    actor.start()

trainer = Trainer(
    context=context,
    shared_context=shared_context,
    replay_buffer=replay_buffer,
)
trainer.start()

games_collector = GamesCollector(
    queue=games_queue,
    stop_event=stop_event,
    replay_buffer=replay_buffer,
    save_frequency=10,
    path=f"{CHECKPOINT_PATH}/replay_buffer.gzip",
)
games_collector.start()

reanalyzer = Reanalyzer(
    context=context,
    shared_context=shared_context,
    stop_event=stop_event,
    replay_buffer=replay_buffer,
)
reanalyzer.start()


def signal_handler(signum, frame):
    tqdm.write("Received signal, stopping threads...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)

try:
    while not stop_event.is_set():
        stop_event.wait(1)
except:
    stop_event.set()
    tqdm.write("Unexpected error! Stopping threads...")
