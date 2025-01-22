import copy
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock(nn.Module):
    """
    Residual block for the representation and dynamics networks.
    """

    def __init__(self, in_channels):
        """
        Residual block for the representation and dynamics networks.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        """

        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))


class RepresentationNetwork(nn.Module):
    """
    The representation network for the MuZero algorithm.

    Processes raw observations from the environment and outputs a hidden state
    representation of the current game state in shape (batch_size, hidden_state_channels, 6, 6).
    """

    def __init__(
        self,
        raw_state_channels: int,
        hidden_state_channels: int,
        res_blocks_per_layer: int = 3,
    ):
        """
        The representation network for the MuZero algorithm.

        Processes raw observations from the environment and outputs a hidden state
        representation of the current game state in shape (batch_size, hidden_state_channels, 6, 6).

        Parameters
        ----------
        raw_state_channels : int
            The number of input channels in the raw observations.
        hidden_state_channels : int
            The number of output channels in the hidden state representation.
        res_blocks_per_layer : int, optional
            The number of residual blocks per layer. Default is 3.
        """
        super(RepresentationNetwork, self).__init__()

        self.conv_1 = nn.Conv2d(
            raw_state_channels,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.res_blocks_1 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.conv_2 = nn.Conv2d(
            in_channels=128,
            out_channels=hidden_state_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.res_blocks_2 = nn.Sequential(
            *[ResidualBlock(hidden_state_channels) for _ in range(res_blocks_per_layer)]
        )

        self.avg_pool_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.res_blocks_3 = nn.Sequential(
            *[ResidualBlock(hidden_state_channels) for _ in range(res_blocks_per_layer)]
        )

        self.avg_pool_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv_1(x))
        x = self.res_blocks_1(x)
        x = F.relu(self.conv_2(x))
        x = self.res_blocks_2(x)
        x = self.avg_pool_1(x)
        x = self.res_blocks_3(x)
        hidden_state = self.avg_pool_2(x)
        return hidden_state


class DynamicsNetwork(nn.Module):
    """
    The dynamics network for the MuZero algorithm.

    Processes the hidden state representation of the current game state and an action
    and outputs the hidden state representation of the next game state and the reward.
    """

    def __init__(
        self,
        hidden_state_channels: int,
        num_actions: int,
        res_blocks_per_layer: int,
        reward_support_size: int,
        hidden_state_size: Tuple[int, int] = (6, 6),
    ):
        """
        The dynamics network for the MuZero algorithm.

        Processes the hidden state representation of the current game state and an action
        and outputs the hidden state representation of the next game state and the reward.

        Parameters
        ----------
        hidden_state_channels : int
            The number of channels in the hidden state representation.
        num_actions : int
            The number of actions in the environment.
        res_blocks_per_layer : int
            The number of residual blocks per layer.
        reward_support_size : int
            The size of the reward support - the number of bins in the reward distribution.
        hidden_state_size : Tuple[int, int], optional
            The size of the hidden state representation. Default is (6, 6).
        """
        super(DynamicsNetwork, self).__init__()

        self.num_actions = num_actions
        self.reward_support_size = reward_support_size

        c = hidden_state_channels + num_actions
        w, h = hidden_state_size

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=hidden_state_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_state_channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_state_channels) for _ in range(res_blocks_per_layer)]
        )

        self.reward_head = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_state_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * h * w, reward_support_size),
        )

    def forward(self, hidden_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the dynamics network.

        Processes the hidden state representation of the current game state
        and an action and outputs the hidden state representation of the next game state and the reward.

        Parameters
        ----------
        hidden_state : Tensor
            The hidden state representation of the current game state.
            Shape (batch_size, hidden_state_channels, width, height).
        action : Tensor
            The action taken in the current game state.
            Shape (batch_size,).

        Returns
        -------
        hidden_state : Tensor
            The hidden state representation of the next game state.
            Shape (batch_size, hidden_state_channels, width, height).
        reward_probabilities : Tensor
            The reward probabilities for the next game state.
            Shape (batch_size, reward_support_size).
        """
        assert hidden_state.shape[0] == action.shape[0]

        b, _, h, w = hidden_state.shape

        # [batch_size, num_actions]
        onehot_action = F.one_hot(action.long(), self.num_actions).to(
            device=hidden_state.device, dtype=torch.float32
        )
        # [batch_size, num_actions, h*w]
        onehot_action = torch.repeat_interleave(onehot_action, repeats=h * w, dim=1)
        # [batch_size, num_actions, h, w]
        onehot_action = torch.reshape(onehot_action, (b, self.num_actions, h, w))

        x = torch.cat([hidden_state, onehot_action], dim=1)
        # [batch_size, hidden_state_channels, h, w]
        hidden_state = self.res_blocks(self.conv_block(x))
        # [batch_size, reward_support_size]
        reward_logits = self.reward_head(hidden_state)

        return hidden_state, reward_logits


class PredictionNetwork(nn.Module):
    """
    The prediction network for the MuZero algorithm.

    Processes the hidden state representation of the current game state and outputs
    policy probabilites and value logits.
    """

    def __init__(
        self,
        hidden_state_channels: int,
        num_actions: int,
        value_support_size: int,
        res_blocks_per_layer: int,
        hidden_state_size: Tuple[int, int] = (6, 6),
    ):
        """
        The prediction network for the MuZero algorithm.

        Processes the hidden state representation of the current game state and outputs
        policy logits and value logits.

        Parameters
        ----------
        hidden_state_channels : int
            The number of channels in the hidden state representation.
        num_actions : int
            The number of actions in the environment.
        value_support_size : int
            The size of the value support - the number of bins in the value distribution.
        res_blocks_per_layer : int
            The number of residual blocks per layer.
        hidden_state_size : Tuple[int, int], optional
            The size of the hidden state representation. Default is (6, 6).
        """

        super(PredictionNetwork, self).__init__()

        self.num_actions = num_actions
        w, h = hidden_state_size

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_state_channels) for _ in range(res_blocks_per_layer)]
        )

        self.policy_net = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_state_channels,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * h * w, num_actions),
        )

        self.value_net = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_state_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * h * w, value_support_size),
        )

    def forward(self, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the prediction network.

        Processes the hidden state representation of the current game state
        and outputs policy logits and value logits.
        """
        features = self.res_blocks(hidden_state)

        policy_logits = self.policy_net(features)
        value_logits = self.value_net(features)

        return policy_logits, value_logits


class MuZeroNetwork:
    """
    The MuZero network.

    Combines the representation, dynamics and prediction networks.
    """

    def __init__(
        self,
        raw_state_channels: int,
        hidden_state_channels: int,
        num_actions: int,
        value_support_size: int,
        reward_support_size: int,
        res_blocks_per_layer: int = 3,
    ):
        """
        The MuZero network.

        Combines the representation, dynamics and prediction networks.

        Parameters
        ----------
        raw_state_channels : int
            The number of input channels in the raw observations.
        hidden_state_channels : int
            The number of output channels in the hidden state representation.
        num_actions : int
            The number of actions in the environment.
        value_support_size : int
            The size of the value support - the number of bins in the value distribution.
        reward_support_size : int
            The size of the reward support - the number of bins in the reward distribution.
        res_blocks_per_layer : int, optional
            The number of residual blocks per layer. Default is 3.
        """
        super(MuZeroNetwork, self).__init__()

        self.total_training_steps = 0
        self._raw_state_channels = raw_state_channels
        self._hidden_state_channels = hidden_state_channels
        self._num_actions = num_actions
        self._value_support_size = value_support_size
        self._reward_support_size = reward_support_size
        self._res_blocks_per_layer = res_blocks_per_layer

        self.representation_network = RepresentationNetwork(
            raw_state_channels=raw_state_channels,
            hidden_state_channels=hidden_state_channels,
            res_blocks_per_layer=res_blocks_per_layer,
        )

        self.dynamics_network = DynamicsNetwork(
            hidden_state_channels=hidden_state_channels,
            num_actions=num_actions,
            res_blocks_per_layer=res_blocks_per_layer,
            reward_support_size=reward_support_size,
        )

        self.prediction_network = PredictionNetwork(
            hidden_state_channels=hidden_state_channels,
            num_actions=num_actions,
            value_support_size=value_support_size,
            res_blocks_per_layer=res_blocks_per_layer,
        )

        self.device = "cpu"

    def initial_inference(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Initial inference through the MuZero network.

        Processes the raw observation and returns the hidden state representation,
        the policy logits and the value logits.

        Parameters
        ----------
        state : Tensor
            The raw observation.
            Shape (batch_size, raw_state_channels, width, height).

        Returns
        -------
        hidden_state : Tensor
            The hidden state representation of the current game state.
            Shape (batch_size, hidden_state_channels, width, height).
        policy_logits : Tensor
            The policy logits for the current game state.
        initial_reward : Tensor
            The initial reward logits for the current game state.
        value_logits : Tensor
            The value logits for the current game state.
        """
        hidden_state = self.to_hidden_state(state)
        policy_logits, value_logits = self.prediction_network(hidden_state)
        initial_reward = self._get_initial_reward_logits(state)
        return hidden_state, policy_logits, initial_reward, value_logits

    def recurrent_inference(
        self, hidden_state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Recurrent inference through the MuZero network.

        Processes the hidden state representation and an action and returns the hidden state
        representation of the next game state, the reward logits and the value logits.

        Parameters
        ----------
        hidden_state : Tensor
            The hidden state representation of the current game state.
            Shape (batch_size, hidden_state_channels, width, height).
        action : Tensor
            The action taken in the current game state.
            Shape (batch_size,).

        Returns
        -------
        hidden_state : Tensor
            The hidden state representation of the next game state.
            Shape (batch_size, hidden_state_channels, width, height).
        reward_logits : Tensor
            The reward logits for the next game state.
        policy_logits : Tensor
            The policy logits for the next game state.
        value_logits : Tensor
            The value logits for the next game state.
        """
        hidden_state, reward_logits = self.next_hidden_state(hidden_state, action)
        policy_logits, value_logits = self.prediction_network(hidden_state)
        return hidden_state, reward_logits, policy_logits, value_logits

    def to_hidden_state(self, x: Tensor) -> Tensor:
        """
        Converts raw observations to hidden state representations.

        Parameters
        ----------
        x : Tensor
            The raw observation.
            Shape (batch_size, raw_state_channels, width, height).

        Returns
        -------
        hidden_state : Tensor
            The hidden state representation of the current game state.
            Shape (batch_size, hidden_state_channels, width, height).
        """
        hidden_state = self.representation_network(x)
        normalize_hidden_state = self.normalize_hidden_state(hidden_state)
        return normalize_hidden_state

    def next_hidden_state(self, hidden_state: Tensor, action: Tensor) -> Tensor:
        """
        Computes the hidden state representation of the next game state.

        Parameters
        ----------
        hidden_state : Tensor
            The hidden state representation of the current game state.
            Shape (batch_size, hidden_state_channels, width, height).
        action : Tensor
            The action taken in the current game state.
            Shape (batch_size,).

        Returns
        -------
        hidden_state : Tensor
            The hidden state representation of the next game state.
            Shape (batch_size, hidden_state_channels, width, height).
        """
        next_hidden_state, reward = self.dynamics_network(hidden_state, action)
        next_hidden_state_normalized = self.normalize_hidden_state(next_hidden_state)
        return next_hidden_state_normalized, reward

    def normalize_hidden_state(self, hidden_state: Tensor) -> Tensor:
        """
        Normalizes the hidden state representation.

        Parameters
        ----------
        hidden_state : Tensor
            The hidden state representation.
            Shape (batch_size, hidden_state_channels, width, height).

        Returns
        -------
        normalized_hidden_state : Tensor
            The normalized hidden state representation.
            Shape (batch_size, hidden_state_channels, width, height).
        """
        min = hidden_state.min(dim=1, keepdim=True).values
        max = hidden_state.max(dim=1, keepdim=True).values

        # 1e-8 is added to the denominator to prevent division by zero
        normalized_state = (hidden_state - min) / (max - min + 1e-8)
        return normalized_state

    def value_to_support(self, logits):
        return self._scalar_to_support(logits, self._value_support_size)

    def reward_to_support(self, logits):
        return self._scalar_to_support(logits, self._reward_support_size)

    def support_to_scalar(self, logits: Tensor):
        probabilities = torch.softmax(logits, dim=1)
        support_max_value = (logits.shape[1] - 1) // 2
        support = (
            torch.arange(-support_max_value, support_max_value + 1)
            .expand(probabilities.shape)
            .float()
            .to(device=probabilities.device)
        )
        x = torch.sum(support * probabilities, dim=1, keepdim=True)
        x = self._invert_support_scaling(x)

        return x

    def _scalar_to_support(self, x: Tensor, support_size: int):
        """
        Under this transformation, each scalar is represented as the
        linear combination of its two adjacent supports, such that the
        original value can be recovered by x = x_low * p_low + x_high * p_high.
        """
        x = self._apply_support_scaling(x)
        support_max_value = (support_size - 1) // 2

        x = torch.clamp(x, -support_max_value, support_max_value)
        floor = x.floor()
        p_high = x - floor
        p_low = 1 - p_high

        # [batch, scalar, support]
        logits = torch.zeros(x.shape[0], x.shape[1], support_size).to(x.device)

        # Fill p_low in support
        x_low_indexes = floor + support_max_value
        logits.scatter_(
            2,
            x_low_indexes.long().unsqueeze(-1),
            p_low.unsqueeze(-1),
        )

        # Fill p_high in support
        x_high_indexes = floor + support_max_value + 1
        p_high = p_high.masked_fill_(2 * support_max_value < x_high_indexes, 0.0)

        x_high_indexes = x_high_indexes.masked_fill_(
            2 * support_max_value < x_high_indexes,
            0.0,
        )

        logits.scatter_(
            2,
            x_high_indexes.long().unsqueeze(-1),
            p_high.unsqueeze(-1),
        )
        return logits

    def _apply_support_scaling(self, x):
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    def _invert_support_scaling(self, x):
        return torch.sign(x) * (
            ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
            ** 2
            - 1
        )

    def _get_initial_reward_logits(self, state: Tensor):
        initial_reward = torch.zeros(
            (state.shape[0], 1),
            requires_grad=True,
        ).to(device=state.device)

        return self._scalar_to_support(
            initial_reward, self._reward_support_size
        ).squeeze()

    def clone(self):
        cloned_network = MuZeroNetwork(
            raw_state_channels=self._raw_state_channels,
            hidden_state_channels=self._hidden_state_channels,
            num_actions=self._num_actions,
            value_support_size=self._value_support_size,
            reward_support_size=self._reward_support_size,
        )

        cloned_network.total_training_steps = self.total_training_steps

        cloned_network.representation_network.load_state_dict(
            copy.deepcopy(self.representation_network.state_dict())
        )
        cloned_network.dynamics_network.load_state_dict(
            copy.deepcopy(self.dynamics_network.state_dict())
        )
        cloned_network.prediction_network.load_state_dict(
            copy.deepcopy(self.prediction_network.state_dict())
        )

        return cloned_network.to(self.device)

    def to(self, device):
        self.device = device
        self.representation_network.to(device)
        self.dynamics_network.to(device)
        self.prediction_network.to(device)
        return self

    def compile(self):
        self.representation_network = torch.jit.script(self.representation_network)
        self.dynamics_network = torch.jit.script(self.dynamics_network)
        self.prediction_network = torch.jit.script(self.prediction_network)

        return self

    def save_checkpoint(self, path):
        torch.save(self, f"{path}/muzero_network.pt")

    def get_weights(self):
        return (
            list(self.representation_network.parameters())
            + list(self.dynamics_network.parameters())
            + list(self.prediction_network.parameters())
        )

    @staticmethod
    def from_checkpoint(path):
        return torch.load(f"{path}/muzero_network.pt", weights_only=False)
