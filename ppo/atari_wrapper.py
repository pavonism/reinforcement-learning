import gymnasium
import cv2
import numpy as np
from collections import deque

class AtariWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, frame_stack=4):
        super(AtariWrapper, self).__init__(env)
        self.env = env
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)  # Frame stack buffer
        self.observation_space = gymnasium.spaces.Box(low=0, high=1.0, shape=(frame_stack, 84, 84), dtype=np.float32)

    def reset(self):
        # Reset the environment and get the initial observation
        observation, _ = self.env.reset()  # Unpack observation and info if tuple
        processed_frame = self.preprocess_frame(observation)
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        return np.stack(self.frames, axis=0)


    def step(self, action):
        observation, reward, done, _, info = self.env.step(action)
        processed_frame = self.preprocess_frame(observation)
        self.frames.append(processed_frame)
        return np.stack(self.frames, axis=0), reward, done, info

    def preprocess_frame(self, frame):
        # Convert to grayscale, resize to 84x84, and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        frame = frame / 255.0  # Normalize to [0, 1]
        return frame
