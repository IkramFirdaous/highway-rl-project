import random
import numpy as np
import torch
import torch.nn as nn
from core.dqn_model import DQN
from core.replay_buffer import ReplayBuffer
from core.config import (
    LR, GAMMA, BUFFER_SIZE, BATCH_SIZE,
    TARGET_UPDATE, OBS_SHAPE, N_ACTIONS
)

class DQNAgent:
    def __init__(self):
        self.gamma           = GAMMA
        self.batch_size      = BATCH_SIZE
        self.target_update   = TARGET_UPDATE
        self.n_actions       = N_ACTIONS
        self.steps           = 0

        self.policy_net = DQN(OBS_SHAPE, N_ACTIONS)
        self.target_net = DQN(OBS_SHAPE, N_ACTIONS)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return self.policy_net(s).argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions)
        rewards_t     = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q  = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())