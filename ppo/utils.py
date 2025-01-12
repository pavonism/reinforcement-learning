import re
import gymnasium
import torch
import wandb
import ale_py

from ppo.atari_wrapper import AtariWrapper


def log_networks_weights_and_biases(actor, critic):
    actor_zero_weights = sum((p == 0).sum().item() for p in actor.parameters())
    critic_zero_weights = sum((p == 0).sum().item() for p in critic.parameters())
    total_actor_weights = sum(p.numel() for p in actor.parameters())
    total_critic_weights = sum(p.numel() for p in critic.parameters())

    actor_sparsity = actor_zero_weights / total_actor_weights
    critic_sparsity = critic_zero_weights / total_critic_weights

    print(f"Actor Sparsity: {actor_sparsity * 100:.2f}%")
    print(f"Critic Sparsity: {critic_sparsity * 100:.2f}%")

    for name, param in actor.named_parameters():
        if param.requires_grad:
            print(f"Actor {name}: mean={param.data.mean().item()}, std={param.data.std().item()}")

    for name, param in critic.named_parameters():
        if param.requires_grad:
            print(f"Critic {name}: mean={param.data.mean().item()}, std={param.data.std().item()}")
            
def init_wandb(starting_episode):
    wandb.init(
        project="ppo-atari",
        config={
            "env_name": "ALE/MsPacman-v5",
            "learning_rate": 2.5e-4,
            "gamma": 0.99,
            "clip_epsilon": 0.1,
            "value_coeff": 1,
            "entropy_coeff": 0.05,
            "num_epochs": 15,
            "batch_size": 256,
        },
        #id="9zfqcg9t",
        #resume="must",
    )
    wandb.config.update({"starting_episode": starting_episode}, allow_val_change=True)
    
def prepare_atari_env(recordings_path) -> AtariWrapper:
    env = gymnasium.make("ALE/MsPacman-v5", render_mode="rgb_array")
    env = gymnasium.wrappers.RecordVideo(
        env,
        recordings_path,
        episode_trigger=lambda episode_id: True
    )
    env = AtariWrapper(env, frame_stack=4, screen_size=84)
    return env
