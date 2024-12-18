import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Increase filters
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 7 * 7, 512),  # Adjust based on input size
        nn.ReLU()
)
        self.policy_head = nn.Linear(512, action_dim)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        if x.dim() == 5:  # Check for 5D input
            x = x.squeeze(1)
        x = self.shared_layers(x)
        return self.policy_head(x), self.value_head(x)

    def act(self, x):
        policy_logits, _ = self.forward(x)
        probs = torch.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, x, actions):
        policy_logits, values = self.forward(x)
        probs = torch.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(values), dist_entropy
