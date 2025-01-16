import gymnasium as gym


class RewardCutWrapper(gym.Wrapper):
    def __init__(self, env, max_reward):
        super(RewardCutWrapper, self).__init__(env)
        self.max_reward = max_reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = 0 if reward > self.max_reward else reward
        return observation, reward, terminated, truncated, info
