from src.policies.common import Policy
import random


class RandomPolicy(Policy):
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def get_action(self, state):
        return random.randint(0, self.n_actions - 1)
