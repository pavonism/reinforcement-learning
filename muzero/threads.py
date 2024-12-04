import logging
import multiprocessing
from multiprocessing.synchronize import Event
import os
import threading

import numpy as np
import torch
from tqdm import tqdm

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
        self._latest_network = network

    def save_game(self, game: Game):
        game.env = None
        self._data_queue.put(game)

    def is_stopped(self) -> bool:
        self._stop_event.is_set()


class ExperienceCollector(threading.Thread):
    def __init__(
        self,
        queue: multiprocessing.Queue,
        replay_buffer: ReplayBuffer,
        save_frequency: int,
        path: str,
    ):
        super().__init__(
            target=self._run,
            args=(queue, replay_buffer, save_frequency, path),
        )

    def _run(
        self,
        queue: multiprocessing.Queue,
        replay_buffer: ReplayBuffer,
        save_frequency: int,
        path: str,
    ):
        logging.info("Started experience collector thread.")

        while True:
            try:
                game = queue.get()
                replay_buffer.save(game)

                if replay_buffer.total_games % save_frequency == 0:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    replay_buffer.save_to_disk(path)
                    logging.info(f"Replay buffer saved to {path}")

            except queue.empty():
                pass
            except EOFError:
                pass


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
            game = self.play_game(context, network)
            shared_context.save_game(game)

    def play_game(self, context: MuZeroContext, network: MuZeroNetwork):
        game = context.new_game(self._actor_id)

        with tqdm(total=context.max_moves, position=self._actor_id) as tqdm_bar:
            while not game.terminal() and len(game.actions) < context.max_moves:
                root = Node(0)
                state = game.get_state(-1)

                hidden_state, policy_logits, _ = network.initial_inference(state)

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
