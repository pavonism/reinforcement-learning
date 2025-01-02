import torch
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(self, actor_critic, input_dim, action_dim, buffer, device, learning_rate=5e-4, gamma=0.99, clip_epsilon=0.2,
                 value_coeff=0.5, entropy_coeff=0.05, num_epochs=25):
        self.policy = actor_critic(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=1e-5)
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

        with torch.no_grad():
            _, state_values = self.policy(states)
            state_values = state_values.squeeze(-1)

        advantages = self.buffer.compute_gae(state_values.cpu().numpy(), gamma=self.gamma, gae_lambda=0.95)

        policy_losses, value_losses, entropies, kl_divs = [], [], [], []

        for _ in range(self.num_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = advantages.to(self.device)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_coeff * (rewards - state_values).pow(2).mean()
            entropy_loss = -self.entropy_coeff * dist_entropy.mean()

            with torch.no_grad():
                kl_div = (ratios - 1 - torch.log(ratios)).mean().item()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(dist_entropy.mean().item())
            kl_divs.append(kl_div)

            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.buffer.clear()
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies), np.mean(kl_divs)



    def select_action(self, state):
        with torch.no_grad():
            if isinstance(state, tuple):
                state = state[0]

            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()

            
            if len(state.shape) == 3 and state.shape[-1] == 3:
                state = np.transpose(state, (2, 0, 1))

            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            action, logprob, entropy = self.policy.act(state)

            self.buffer.states.append(state.cpu().numpy())
            self.buffer.actions.append(action.item())
            self.buffer.logprobs.append(logprob.item())

            return action.item()

