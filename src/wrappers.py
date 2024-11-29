import gymnasium as gym


class TerminateOnLifeLossWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return observation

    def step(self, action):
        observation, reward, done, info, *_ = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives:
            done = True

        return observation, reward, done, info
