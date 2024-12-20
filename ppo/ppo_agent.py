import torch
import torch.optim as optim
import numpy as np


class PPO:
    def __init__(
        self,
        actor_critic,
        input_dim,
        action_dim,
        buffer,
        device,
        log_path,
        learning_rate=3e-5,
        gamma=0.99,
        clip_epsilon=0.15,
        value_coeff=0.5,
        entropy_coeff=0.05,
        num_epochs=20,
    ):
        self.policy = actor_critic(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = buffer
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.num_epochs = num_epochs
        self.log_path = log_path

    def update(self):
        rewards = self.buffer.compute_rewards_to_go(self.gamma)
        states = torch.tensor(np.stack(self.buffer.states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(self.buffer.actions, dtype=torch.int64).to(self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32).to(
            self.device
        )

        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(self.device)

        with torch.no_grad():
            _, old_values = self.policy.forward(states)

        dataset_size = len(self.buffer.states)
        mini_batch_size = 512  # or 256, depending on your GPU memory

        with open(self.log_path, "a") as log_file:
            for _ in range(self.num_epochs):
                permutation = torch.randperm(dataset_size)

                for batch_start in range(0, dataset_size, mini_batch_size):
                    batch_end = batch_start + mini_batch_size
                    batch_indices = permutation[batch_start:batch_end]

                    # Get mini-batch data
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_logprobs = old_logprobs[batch_indices]
                    batch_rewards = rewards[batch_indices]
                    batch_old_values = old_values[batch_indices]

                    logprobs, state_values, dist_entropy = self.policy.evaluate(
                        batch_states, batch_actions
                    )
                    ratios = torch.exp(logprobs - batch_old_logprobs.detach())

                    state_values = state_values.to(self.device)
                    advantages = batch_rewards - state_values.detach()
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Compute losses
                    surr1 = ratios * advantages
                    surr2 = (
                        torch.clamp(
                            ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                        )
                        * advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss with clipping
                    value_pred_clipped = batch_old_values + torch.clamp(
                        state_values - batch_old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_losses = (state_values - batch_rewards).pow(2)
                    value_losses_clipped = (value_pred_clipped - batch_rewards).pow(2)
                    value_loss = (
                        self.value_coeff
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )

                    entropy_loss = -self.entropy_coeff * dist_entropy.mean()

                    loss = policy_loss + value_loss + entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), max_norm=0.5
                    )
                    self.optimizer.step()
                    log_file.write(f"{loss.item()}\n")

            log_file.flush()

        self.buffer.clear()
        self.entropy_coeff = max(0.01, self.entropy_coeff * 0.92)

    def select_action(self, state):
        with torch.no_grad():
            if isinstance(state, tuple):
                state = state[0]  # Extract observation if state is a tuple

            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()

            # Add batch dimension if needed
            if len(state.shape) == 3 and state.shape[-1] == 3:
                state = np.transpose(state, (2, 0, 1))

            state = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            action, logprob, entropy = self.policy.act(state)

            # Store CPU versions in buffer
            self.buffer.states.append(state.cpu().numpy())
            self.buffer.actions.append(action.item())
            self.buffer.logprobs.append(logprob.item())

            return action.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
