import torch


class PPOContext:
    def __init__(
        self,
        update_interval=4096,
        learning_rate=5e-5,
        gamma=0.99,
        clip_epsilon=0.1,
        value_coeff=1,
        entropy_coeff=0.02,
        num_epochs=15,
        total_timesteps=1e7,
    ):
        self.update_interval = update_interval
        self.learning_rate = learning_rate
        self.weight_decay = 1e-5
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.num_epochs = num_epochs
        self.total_timesteps = total_timesteps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
