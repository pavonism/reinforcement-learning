import torch
import torch.optim as optim
import numpy as np

from ppo.actor_critic import ActorCritic
from ppo.buffer import RolloutBuffer
from ppo.context import PPOContext


class PPO:
    def __init__(
        self,
        context: PPOContext,
        actor_critic: ActorCritic,
        buffer: RolloutBuffer,
    ):
        self.policy = actor_critic.to(context.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=context.learning_rate,
            weight_decay=context.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.5,
            total_iters=int(context.total_timesteps / 4096),  # Decay over all updates
        )

        # Entropy coefficient scheduling
        self.initial_entropy_coeff = context.entropy_coeff
        self.current_entropy_coeff = context.entropy_coeff
        self.min_entropy_coeff = 0.02
        # Decay over 75% of training
        self.entropy_decay_steps = int(context.total_timesteps * 0.75)
        self.entropy_decay_rate = (
            context.entropy_coeff - self.min_entropy_coeff
        ) / self.entropy_decay_steps

        self.context = context
        self.buffer = buffer
        self.total_steps = 0

    def update(self, timestep):
        self._update_entropy_coeff(timestep)

        rewards = self.buffer.compute_rewards_to_go(self.context.gamma)

        states = torch.tensor(
            np.stack(self.buffer.states),
            dtype=torch.float32,
        ).to(self.context.device)

        actions = torch.tensor(
            self.buffer.actions,
            dtype=torch.int64,
        ).to(self.context.device)

        old_logprobs = torch.tensor(
            self.buffer.logprobs,
            dtype=torch.float32,
        ).to(self.context.device)

        with torch.no_grad():
            _, state_values = self.policy(states)
            state_values = state_values.squeeze(-1)

        advantages = self.buffer.compute_gae(
            state_values.cpu().numpy(),
            gamma=self.context.gamma,
            gae_lambda=0.95,
        )

        policy_losses, value_losses, entropies, kl_divs = [], [], [], []

        for _ in range(self.context.num_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = advantages.to(self.context.device)

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(
                    ratios,
                    1 - self.context.clip_epsilon,
                    1 + self.context.clip_epsilon,
                )
                * advantages
            )

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (
                self.context.value_coeff * (rewards - state_values).pow(2).mean()
            )
            entropy_loss = -self.current_entropy_coeff * dist_entropy.mean()

            with torch.no_grad():
                kl_div = (ratios - 1 - torch.log(ratios)).mean().item()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(dist_entropy.mean().item())
            kl_divs.append(kl_div)

            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.scheduler.step()

        self.buffer.clear()
        current_lr = self.scheduler.get_last_lr()[0]

        return (
            np.mean(policy_losses),
            np.mean(value_losses),
            np.mean(entropies),
            np.mean(kl_divs),
            current_lr,
            self.current_entropy_coeff,
        )

    def _update_entropy_coeff(self, timestep):
        """Update entropy coefficient using linear decay"""
        if timestep < self.entropy_decay_steps:
            self.current_entropy_coeff = max(
                self.initial_entropy_coeff - (self.entropy_decay_rate * timestep),
                self.min_entropy_coeff,
            )

    def select_action(self, state):
        with torch.no_grad():
            if isinstance(state, tuple):
                state = state[0]

            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()

            if len(state.shape) == 3 and state.shape[-1] == 3:
                state = np.transpose(state, (2, 0, 1))

            state = (
                torch.tensor(state, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.context.device)
            )

            action, logprob, _ = self.policy.act(state)

            self.buffer.states.append(state.cpu().numpy())
            self.buffer.actions.append(action.item())
            self.buffer.logprobs.append(logprob.item())

            return action.item()
