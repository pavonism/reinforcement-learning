import gymnasium
import cv2
import numpy as np
from collections import deque

class AtariWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, frame_stack=4, screen_size=84):
        super(AtariWrapper, self).__init__(env)
        self.frame_stack = frame_stack
        self.screen_size = screen_size
        self.frames = deque([], maxlen=frame_stack)
        
        self.observation_space = gymnasium.spaces.Box(
            low=0, 
            high=1.0,
            shape=(frame_stack, screen_size, screen_size),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        processed_frame = self.preprocess_frame(observation)
        
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        
        stacked_frames = np.concatenate(self.frames, axis=0)
        return stacked_frames, info


    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        processed_frame = self.preprocess_frame(observation)
        self.frames.append(processed_frame)
        
        stacked_frames = np.concatenate(self.frames, axis=0)
        return stacked_frames, reward, terminated, truncated, info


    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.screen_size, self.screen_size))
        frame = frame.astype(np.float32) / 255.0
        return np.expand_dims(frame, axis=0)


