import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    DQN architecture inspired by Mnih et al. 2015, adapted for
    Kinematics observations (10 vehicles x 5 features).
    Conv layers extract spatial relationships between vehicles;
    a fully-connected head estimates Q-values.
    No BatchNorm: absent from the original paper and unstable in RL.
    """
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        h, w = obs_shape  # (10, 5)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            conv_out_dim = self.conv(dummy).flatten(start_dim=1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.head(x)