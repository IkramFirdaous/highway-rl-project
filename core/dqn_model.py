import numpy as np
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        obs_size = int(np.prod(obs_shape))
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))