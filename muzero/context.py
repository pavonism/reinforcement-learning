import collections
from typing import Callable, Optional
from gymnasium import Env

from muzero.game import Game

KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -float("inf")
        self.minimum = known_bounds.min if known_bounds else float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MuZeroContext(object):
    """MuZero Hyperparameters."""

    def __init__(
        self,
        n_actions: int,
        max_moves: int,
        discount: float,
        dirichlet_alpha: float,
        num_simulations: int,
        batch_size: int,
        td_steps: int,
        num_actors: int,
        lr_init: float,
        lr_decay_steps: float,
        env_factory: Callable[[int], Env],
        checkpoint_path: str,
        n_states_representation: int = 32,
        n_actions_representation: int = 32,
        value_loss_weight: float = 1.0,
        train_device: str = "cpu",
        act_device: str = "cpu",
        known_bounds: Optional[KnownBounds] = None,
    ):
        ### Self-Play
        self.n_actions = n_actions
        self.num_actors = num_actors
        self.act_device = act_device

        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        # Environment
        self._env_factory = env_factory
        self._envs = [env_factory(i) for i in range(num_actors)]

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(250)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps
        self.value_loss_weight = value_loss_weight
        self.train_device = train_device

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        ### Checkpoint path.
        self.checkpoint_path = checkpoint_path

        ### Representation function parameters
        self.n_states_representation = n_states_representation
        self.n_actions_representation = n_actions_representation

    def new_game(self, actor_id: int):
        return Game(
            self._envs[actor_id],
            self.n_actions,
            self.discount,
            self.act_device,
        )

    def visit_softmax_temperature(
        self,
        num_moves: int,
        training_steps: int,
    ):
        """
        Determines the level of exploration in the search tree.

        Args:
            num_moves: The number of moves that have been made in the game.
            training_steps: The number of training steps that have been executed.
        """

        # In the paper, they use a step function to change the exploration.
        # Number of moves was not used in the end.
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25
