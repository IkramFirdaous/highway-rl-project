import torch
import torch.nn as nn
from core.dqn_agent import DQNAgent


class DDQNAgent(DQNAgent):
    """
    Double DQN (van Hasselt et al. 2016).
    Seul changement vs DQN : le reseau policy choisit l'action,
    le reseau target evalue sa valeur — evite la surestimation des Q-values.
    """

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q       = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets      = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
