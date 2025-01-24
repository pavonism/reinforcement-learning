from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


class ActorNet(nn.Module):
    """Actor network for the PPO algorithm."""

    def __init__(self, input_dim: list, action_dim: int):
        """
        Initialize the actor network.

        Args:
            input_dim (list): Dimension of the input tensors.
            action_dim (int): Dimension of the action space.
        """

        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out_size(input_dim)

        self.fc_net = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1),
        )

    def _get_conv_out_size(self, input_dim: list) -> int:
        x = torch.zeros(1, *input_dim)
        x = self.conv_net(x)
        return x.shape[1]

    def forward(self, x):
        x = self.conv_net(x)
        return self.fc_net(x)


class CriticNet(nn.Module):
    """Critic network for the PPO algorithm."""

    def __init__(self, input_dim: list):
        """
        Initialize the critic network.

        Args:
            input_dim (list): Dimension of the input tensors.
        """

        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out_size(input_dim)

        self.fc_net = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def _get_conv_out_size(self, input_dim: list) -> int:
        x = torch.zeros(1, *input_dim)
        x = self.conv_net(x)
        return x.shape[1]

    def forward(self, x):
        x = self.conv_net(x)
        return self.fc_net(x)


class ActorCritic(nn.Module):
    """Actor-Critic network for the PPO algorithm."""

    def __init__(self, input_dim: list, action_dim: int):
        super().__init__()
        self.actor = ActorNet(input_dim, action_dim)
        self.critic = CriticNet(input_dim)
        self.apply(self._weights_init)

    def forward(self, x: torch.Tensor):
        if x.dim() == 5:
            x = x.squeeze(1)

        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def act(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        action_probs, _ = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, x: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        action_probs, values = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy
