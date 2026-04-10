import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        h, w = obs_shape  # (10, 5)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            conv_out_dim = self.conv(dummy).flatten(start_dim=1).shape[1]
        self.head = nn.Linear(conv_out_dim, n_actions)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.head(x)