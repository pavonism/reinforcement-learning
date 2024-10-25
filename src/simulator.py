from gymnasium import Env
from src.policies.common import Policy


class Simulator:
    def __init__(self, env: Env, policy: Policy, output_path: str) -> None:
        self.env = env
        self.policy = policy
        self.output_path = output_path

    def run(self, max_steps: int = 1000) -> float:
        state = self.env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = self.policy.get_action(state)
            state, reward, done, _, _ = self.env.step(action)
            total_reward += reward

            if done:
                break

        state = self.env.reset()
        return total_reward
