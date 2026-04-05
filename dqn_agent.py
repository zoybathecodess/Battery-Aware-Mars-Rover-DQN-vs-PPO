import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=50000)

        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.992

        self.learning_rate = 0.0005

        self.model        = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate, eps=1e-4)
        self.criterion = nn.SmoothL1Loss()

        self.train_step         = 0
        self.target_update_freq = 500

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states      = torch.FloatTensor(np.array([m[0] for m in minibatch]))
        actions     = torch.LongTensor( [m[1] for m in minibatch])
        rewards     = torch.FloatTensor([m[2] for m in minibatch])
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch]))
        dones       = torch.FloatTensor([m[4] for m in minibatch])

        # Double DQN
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, best_actions).squeeze(1)

        target_q  = rewards + self.gamma * next_q * (1 - dones)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
