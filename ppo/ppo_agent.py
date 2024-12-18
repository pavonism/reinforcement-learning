import torch
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(self, actor_critic, input_dim, action_dim, buffer, device, learning_rate=3e-4, gamma=0.99, clip_epsilon=0.2,
                 value_coeff=0.5, entropy_coeff=0.05, num_epochs=20):
        self.policy = actor_critic(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = buffer
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.num_epochs = num_epochs

    def update(self):
        rewards = self.buffer.compute_rewards_to_go(self.gamma)
        states = torch.tensor(np.stack(self.buffer.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.int64).to(self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32).to(self.device)

        for _ in range(self.num_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            loss = -torch.min(surr1, surr2) + \
                   self.value_coeff * (rewards - state_values).pow(2) - \
                   self.entropy_coeff * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.buffer.clear()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, logprob, entropy = self.policy.act(state)
        self.buffer.states.append(state.cpu().numpy())
        self.buffer.actions.append(action.item())
        self.buffer.logprobs.append(logprob.item())
        return action.item()
