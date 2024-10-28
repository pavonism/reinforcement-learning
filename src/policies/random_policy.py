from gymnasium import Env
from tqdm import tqdm
from src.policies.common import Policy
import random


class RandomPolicy(Policy):
    def __init__(
        self,
        env: Env,
        seed: int = 0,
    ):
        self.__env = env
        self.__n_actions = env.action_space.n
        random.seed(seed)

    def train(self, n_episodes: int = 1, max_steps: int = 1_000) -> float:
        raise NotImplementedError("Random policy does not require training")

    def play(self, n_episodes: int = 1, max_steps: int = 1_000) -> float:
        self.__env.reset()
        max_reward = 0

        for _ in tqdm(range(n_episodes), desc="Running episodes"):
            self.__env.reset()
            total_reward = 0

            for _ in range(max_steps):
                action = self.get_action()
                _, reward, done, _, _ = self.__env.step(action)
                total_reward += reward

                if done:
                    break

            max_reward = max(max_reward, total_reward)

        self.__env.reset()
        return max_reward

    def get_action(self):
        return random.randint(0, self.__n_actions - 1)
