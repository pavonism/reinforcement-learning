import os
from queue import Empty, Queue
import threading
import time
import traceback

import numpy as np
import torch
from torch.functional import F
from tqdm import tqdm
import wandb

from muzero.context import MuZeroContext
from muzero.game import Game
from muzero.networks import MuZeroNetwork
from muzero.replay import BatchedExperiences, ReplayBuffer
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
        games_queue: Queue,
        stop_event: threading.Event,
    ):
        self._latest_network = network
        self._data_queue = games_queue
        self._stop_event = stop_event

    def get_latest_network(self):
        return self._latest_network

    def set_network(self, network: MuZeroNetwork):
        self._latest_network = network

    def save_game(self, game: Game):
        game.env = None
        self._data_queue.put(game)

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()


class GamesCollector(threading.Thread):
    def __init__(
        self,
        queue: Queue,
        stop_event: threading.Event,
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
        queue: Queue,
        stop_event: threading.Event,
        replay_buffer: ReplayBuffer,
        save_frequency: int,
        path: str,
    ):
        tqdm.write("Started games collector thread.")

        while not stop_event.is_set():
            try:
                game = queue.get(timeout=5)
                replay_buffer.save(game)

                if (
                    replay_buffer.total_games % save_frequency == 0
                    or replay_buffer.total_games < save_frequency
                ):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    replay_buffer.save_to_disk()
                    tqdm.write(f"Replay buffer saved to {path}")

            except Empty:
                continue
            except EOFError:
                pass
            except Exception as e:
                tqdm.write("Games collector error:")
                tqdm.write(str(e))
                tqdm.write(traceback.format_exc())
                stop_event.set()

        tqdm.write("Games collector stopped.")


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
        tqdm.write(f"Started actor {self._actor_id} thread.")

        try:
            while not shared_context.is_stopped():
                network = shared_context.get_latest_network()
                game = self.play_game(context, shared_context, network)

                if not game.terminal():
                    break

                wandb.log(
                    {
                        f"game_length_actor_{self._actor_id}": len(game.actions),
                        f"total_reward_actor_{self._actor_id}": sum(game.rewards),
                    }
                )
                shared_context.save_game(game)
        except Exception as e:
            tqdm.write(f"Actor {self._actor_id} error:")
            tqdm.write(str(e))
            tqdm.write(traceback.format_exc())
            shared_context.stop()

        tqdm.write(f"Actor {self._actor_id} stopped.")

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
                state = game.get_state(
                    len(game.states) - 1,
                    context.n_states_representation,
                    context.n_actions_representation,
                )

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
            training_steps=network.total_training_steps,
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
        tqdm.write("Started trainer thread.")

        network = shared_context.get_latest_network().clone().to(context.train_device)

        optimizer = torch.optim.SGD(
            network.get_weights(),
            lr=context.lr_init,
            momentum=context.momentum,
            weight_decay=context.weight_decay,
        )

        try:
            with tqdm(
                total=context.training_steps,
                position=0,
                desc="Trainer",
                initial=network.total_training_steps,
            ) as p_bar:
                for i in range(network.total_training_steps, context.training_steps):
                    if shared_context.is_stopped():
                        break

                    if replay_buffer.total_games == 0:
                        time.sleep(1)
                        continue

                    self._update_lr(optimizer, context, i)

                    if i % context.checkpoint_interval == 0:
                        self._save_network(context, network)

                    batch = replay_buffer.sample(
                        context.batch_size,
                        context.train_device,
                        context.n_states_representation,
                        context.n_actions_representation,
                    )
                    self.train_network(
                        context, replay_buffer, optimizer, network, batch
                    )
                    network.total_training_steps += 1

                    shared_context.set_network(network.clone().to(context.act_device))
                    p_bar.update(1)

                self._save_network(context, network)
        except Exception as e:
            tqdm.write("Trainer error:")
            tqdm.write(str(e))
            tqdm.write(traceback.format_exc())
            shared_context.stop()

        tqdm.write("Trainer stopped.")

    def train_network(
        self,
        context: MuZeroContext,
        replay_buffer: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
        network: MuZeroNetwork,
        batch: BatchedExperiences,
    ):
        (
            state_indexes,
            states,
            gradient_scales,
            actions,
            target_values,
            target_rewards,
            target_policies,
            corrections,
        ) = batch

        hidden_states, policy_logits, reward_logits, value_logits = (
            network.initial_inference(states)
        )

        priorities = np.zeros(target_values.shape)

        predictions = [
            (
                1.0,
                value_logits,
                reward_logits,
                policy_logits,
            )
        ]

        for i in range(0, actions.shape[1]):
            hidden_states, reward_logits, policy_logits, value_logits = (
                network.recurrent_inference(hidden_states, actions[:, i])
            )

            predictions.append(
                (gradient_scales[i], value_logits, reward_logits, policy_logits)
            )

            hidden_states.register_hook(lambda grad: grad * 0.5)

        value_loss, reward_loss, policy_loss = 0, 0, 0

        for i in range(len(predictions)):
            gradient_scale, value, reward, policy_logits = predictions[i]
            target_value, target_reward, target_policy = (
                target_values[:, i],
                target_rewards[:, i],
                target_policies[:, i],
            )

            value_as_scalar = (
                network.support_to_scalar(value).detach().cpu().squeeze().numpy()
            )
            priorities[:, i] = np.abs(
                target_value.detach().cpu().numpy() - value_as_scalar
            )

            target_value = network.value_to_support(target_value.unsqueeze(1)).squeeze()
            target_reward = network.reward_to_support(
                target_reward.unsqueeze(1)
            ).squeeze()

            current_value_loss = F.cross_entropy(
                value,
                target_value,
                reduction="none",
            )
            current_reward_loss = F.cross_entropy(
                reward,
                target_reward,
                reduction="none",
            )
            current_policy_loss = F.cross_entropy(
                policy_logits,
                target_policy.squeeze(),
                reduction="none",
            )

            current_value_loss.register_hook(lambda grad: grad * gradient_scale)
            current_reward_loss.register_hook(lambda grad: grad * gradient_scale)
            current_policy_loss.register_hook(lambda grad: grad * gradient_scale)

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

        value_loss *= corrections * context.value_loss_weight
        reward_loss *= corrections
        policy_loss *= corrections

        loss_batch = value_loss + reward_loss + policy_loss
        loss = loss_batch.mean()  # Original pseudocode uses sum

        wandb.log(
            {
                "value_loss": value_loss.mean(),
                "reward_loss": reward_loss.mean(),
                "policy_loss": policy_loss.mean(),
            }
        )

        wandb.log({"total_loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        replay_buffer.update_priorities(state_indexes, priorities)

    def _save_network(self, context: MuZeroContext, network: MuZeroNetwork):
        network.save_checkpoint(context.checkpoint_path)
        tqdm.write("Network saved.")

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
