import logging
import multiprocessing
from multiprocessing.synchronize import Event
import os
import threading
import time

import numpy as np
import torch
from torch import Tensor
from torch.functional import F
from tqdm import tqdm
import wandb

from muzero.context import MuZeroContext
from muzero.game import Game
from muzero.networks import MuZeroNetwork
from muzero.replay import ReplayBuffer
from muzero.tree_search import (
    Node,
    add_exploration_noise,
    expand_node,
    run_mcts,
)


class SharedContext:
    def __init__(
        self,
        network: MuZeroNetwork,
        games_queue: multiprocessing.Queue,
        stop_event: Event,
    ):
        self._latest_network = network
        self._data_queue = games_queue
        self._stop_event = stop_event

    def get_latest_network(self):
        return self._latest_network

    def set_network(self, network: MuZeroNetwork):
        self._latest_network = network.clone()

    def save_game(self, game: Game):
        game.env = None
        self._data_queue.put(game)

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()


class GamesCollector(threading.Thread):
    def __init__(
        self,
        queue: multiprocessing.Queue,
        stop_event: Event,
        replay_buffer: ReplayBuffer,
        save_frequency: int,
        path: str,
    ):
        super().__init__(
            target=self._run,
            args=(queue, stop_event, replay_buffer, save_frequency, path),
        )

    def _run(
        self,
        queue: multiprocessing.Queue,
        stop_event: Event,
        replay_buffer: ReplayBuffer,
        save_frequency: int,
        path: str,
    ):
        logging.info("Started games collector thread.")

        while not stop_event.is_set():
            try:
                game = queue.get()
                replay_buffer.save(game)

                # if replay_buffer.total_games % save_frequency == 0:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                replay_buffer.save_to_disk(path)
                logging.info(f"Replay buffer saved to {path}")

            except queue.empty():
                pass
            except EOFError:
                pass

        logging.info("Games collector stopped.")


class Actor(threading.Thread):
    def __init__(
        self,
        actor_id: int,
        context: MuZeroContext,
        shared_context: SharedContext,
    ):
        self._actor_id = actor_id

        super().__init__(
            target=self._run,
            args=(context, shared_context),
        )

    @torch.no_grad()
    def _run(
        self,
        context: MuZeroContext,
        shared_context: SharedContext,
    ):
        logging.info(f"Started actor {self._actor_id} thread.")

        while not shared_context.is_stopped():
            network = shared_context.get_latest_network()
            game = self.play_game(context, shared_context, network)
            wandb.log(
                {
                    f"game_length_actor_{self._actor_id}": len(game.actions),
                    f"total_reward_actor_{self._actor_id}": sum(game.rewards),
                }
            )
            shared_context.save_game(game)

        logging.info(f"Actor {self._actor_id} stopped.")

    def play_game(
        self,
        context: MuZeroContext,
        shared_context: SharedContext,
        network: MuZeroNetwork,
    ):
        game = context.new_game(self._actor_id)

        # First position is reserved for the trainer.
        with tqdm(
            total=context.max_moves,
            position=self._actor_id + 1,
            leave=False,
            desc=f"Actor {self._actor_id}",
        ) as tqdm_bar:
            while (
                not game.terminal()
                and len(game.actions) < context.max_moves
                and not shared_context.is_stopped()
            ):
                root = Node(0)
                state = game.get_state(-1)

                hidden_state, policy_logits, *_ = network.initial_inference(state)

                expand_node(
                    root,
                    hidden_state,
                    policy_logits,
                    reward=0,
                )

                add_exploration_noise(context, root)

                run_mcts(context, root, game.get_action_history(), network)

                action = self.select_action(
                    context,
                    len(game.get_action_history()),
                    root,
                    network,
                )

                game.apply_action(action)
                game.store_search_statistics(root)

                tqdm_bar.update(1)

        return game

    def select_action(
        self, context: MuZeroContext, num_moves: int, node: Node, network: MuZeroNetwork
    ):
        t = context.visit_softmax_temperature(
            num_moves=num_moves,
            training_steps=network.get_total_training_steps(),
        )

        return self.select_action_with_temperature(t, node)

    def select_action_with_temperature(self, temperature: float, node: Node):
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action


class Trainer(threading.Thread):
    def __init__(
        self,
        context: MuZeroContext,
        shared_context: SharedContext,
        replay_buffer: ReplayBuffer,
    ):
        super().__init__(
            target=self._run,
            args=(context, shared_context, replay_buffer),
        )

    def _run(
        self,
        context: MuZeroContext,
        shared_context: SharedContext,
        replay_buffer: ReplayBuffer,
    ):
        logging.info("Started trainer thread.")

        network = shared_context.get_latest_network().clone()
        total_training_steps = network.get_total_training_steps()

        optimizer = torch.optim.SGD(
            network.get_weights(),
            lr=context.lr_init,
            momentum=context.momentum,
            weight_decay=context.weight_decay,
        )

        with tqdm(total=context.training_steps, position=0, desc="Trainer") as p_bar:
            for i in range(total_training_steps, context.training_steps):
                if shared_context.is_stopped():
                    break

                if replay_buffer.total_games == 0:
                    time.sleep(1)
                    continue

                self._update_lr(optimizer, context, i)

                if i % context.checkpoint_interval == 0:
                    self._save_network(context, network)

                batch = replay_buffer.sample(
                    context.num_unroll_steps,
                    context.td_steps,
                    context.batch_size,
                )
                self.train_network(context, optimizer, network, batch)

                shared_context.set_network(network)
                p_bar.update(1)

            self._save_network(context, network)

        logging.info("Trainer stopped.")

    def train_network(
        self,
        context: MuZeroContext,
        optimizer: torch.optim.Optimizer,
        network: MuZeroNetwork,
        batch,
    ):
        loss = 0
        for state, actions, targets in batch:
            hidden_state, policy_logits, reward_logits, value_logits = (
                network.initial_inference(state)
            )
            predictions = [(1.0, value_logits, reward_logits, policy_logits)]

            for action in actions:
                hidden_state, reward_logits, policy_logits, value_logits = (
                    network.recurrent_inference(
                        hidden_state,
                        Tensor([action]).unsqueeze(0).to(hidden_state.device),
                    )
                )

                predictions.append(
                    (1.0 / len(actions), value_logits, reward_logits, policy_logits)
                )

                hidden_state.register_hook(lambda grad: grad * 0.5)

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                value_loss = F.cross_entropy(
                    value.squeeze(),
                    network.value_to_support(Tensor([[target_value]])).squeeze(),
                )

                reward_loss = F.cross_entropy(
                    reward.squeeze(),
                    network.reward_to_support(Tensor([[target_reward]])).squeeze(),
                )

                policy_loss = F.cross_entropy(
                    policy_logits.squeeze(), Tensor(target_policy)
                )

                loss += value_loss + reward_loss + policy_loss

                wandb.log(
                    {
                        "value_loss": value_loss.item(),
                        "reward_loss": reward_loss.item(),
                        "policy_loss": policy_loss.item(),
                    }
                )

        loss.register_hook(lambda grad: grad * gradient_scale)
        wandb.log({"total_loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _save_network(self, context: MuZeroContext, network: MuZeroNetwork):
        network.save_checkpoint(context.checkpoint_path)
        logging.info("Network saved.")

    def _update_lr(
        self,
        optimizer: torch.optim.Optimizer,
        context: MuZeroContext,
        total_training_steps: int,
    ):
        learning_rate = context.lr_init * context.lr_decay_rate ** (
            total_training_steps / context.lr_decay_steps
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        wandb.log({"learning_rate": learning_rate})
