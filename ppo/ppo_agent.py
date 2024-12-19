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
            rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(self.device)
            state_values = state_values.to(self.device)
            advantages = rewards - state_values.detach()
            
            # Compute losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_coeff * (rewards - state_values).pow(2).mean()
            entropy_loss = -self.entropy_coeff * dist_entropy.mean()

            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()

    def select_action(self, state):
        with torch.no_grad():
            if isinstance(state, tuple):
                state = state[0]  # Extract observation if state is a tuple

            # Handle different input types
            if isinstance(state, torch.Tensor):
                if state.device != self.device:
                    state = state.to(self.device)
            else:
                state = np.array(state)
                # Convert channel-last to channel-first if necessary
                if len(state.shape) == 3 and state.shape[-1] == 3:
                    state = np.transpose(state, (2, 0, 1))
                state = torch.tensor(state, dtype=torch.float32).to(self.device)

            # Add batch dimension if needed
            if len(state.shape) == 3:
                state = state.unsqueeze(0)

            action, logprob, entropy = self.policy.act(state)

            # Store CPU versions in buffer
            self.buffer.states.append(state.cpu().numpy())
            self.buffer.actions.append(action.item())
            self.buffer.logprobs.append(logprob.item())

            return action.item()
