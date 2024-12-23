import logging
from queue import Queue
import threading
import os
import signal

import gymnasium
import ale_py
import torch
import wandb

from muzero.context import MuZeroContext
from muzero.networks import MuZeroNetwork
from muzero.replay import ReplayBuffer
from muzero.threads import Actor, GamesCollector, SharedContext, Trainer

CHECKPOINT_PATH = "checkpoints/muzero_gpu"
games_queue = Queue()
stop_event = threading.Event()
replay_buffer = ReplayBuffer(capacity=1500)
train_device = "cuda" if torch.cuda.is_available() else "cpu"
actor_device = "cpu"
print("Training on", train_device)
print("Acting on", actor_device)

if os.path.exists(f"{CHECKPOINT_PATH}/replay_buffer.gzip"):
    replay_buffer.load_from_disk(f"{CHECKPOINT_PATH}/replay_buffer.gzip")

gymnasium.register_envs(ale_py)
wandb.login()
wandb.init(project="muzero")
logging.basicConfig(level=logging.INFO)
torch.set_printoptions(profile="full")


def env_factory(actor_id: int):
    return gymnasium.wrappers.AtariPreprocessing(
        gymnasium.wrappers.RecordVideo(
            gymnasium.make("ALE/MsPacman-v5", render_mode="rgb_array"),
            f"{CHECKPOINT_PATH}/recordings/{actor_id}",
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
    batch_size=128,
    td_steps=10,
    num_actors=2,
    lr_init=0.05,
    lr_decay_steps=350e3,
    env_factory=env_factory,
    checkpoint_path=CHECKPOINT_PATH,
    train_device=train_device,
)

network = (
    MuZeroNetwork.from_checkpoint(CHECKPOINT_PATH)
    if os.path.exists(f"{CHECKPOINT_PATH}/muzero_network.pt")
    else MuZeroNetwork(
        raw_state_channels=3,
        hidden_state_channels=128,
        num_actions=context.n_actions,
        value_support_size=601,
        reward_support_size=601,
    )
).to(actor_device)

shared_context = SharedContext(
    games_queue=games_queue,
    network=network,
    stop_event=stop_event,
)

games_collector = GamesCollector(
    queue=games_queue,
    stop_event=stop_event,
    replay_buffer=replay_buffer,
    save_frequency=50,
    path=f"{CHECKPOINT_PATH}/replay_buffer.gzip",
)
games_collector.start()

trainer = Trainer(
    context=context,
    shared_context=shared_context,
    replay_buffer=replay_buffer,
)
trainer.start()

for actor_id in range(context.num_actors):
    actor = Actor(
        actor_id=actor_id,
        context=context,
        shared_context=shared_context,
    )
    actor.start()


def signal_handler(signum, frame):
    print("Received signal, stopping threads...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)
while not stop_event.is_set():
    stop_event.wait(1)
print("All threads stopped.")
