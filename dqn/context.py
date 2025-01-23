import torch


class Context:
    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.0001,
        epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        discount_factor: float = 0.99,
        target_update_frequency: int = 10_000,
        save_frequency_in_episodes: int = 100,
        batch_size: int = 128,
        stack_frame_size=4,
        priority_replay: bool = False,
        replay_buffer_size: int = 800_000,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.target_update_frequency = target_update_frequency
        self.save_frequency_in_episodes = save_frequency_in_episodes
        self.batch_size = batch_size
        self.stack_frame = stack_frame_size
        self.priority_replay = priority_replay
        self.replay_buffer_size = replay_buffer_size
        self.device = device
