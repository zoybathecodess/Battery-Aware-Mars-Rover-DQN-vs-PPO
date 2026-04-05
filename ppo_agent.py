import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()

        # FIX 1: wider network — 128 is too small for a 17-dim battery-aware state
        self.shared = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),          # FIX 2: Tanh over ReLU for PPO — smoother gradients,
            nn.Linear(256, 256),# less likely to produce dead neurons in policy head
            nn.Tanh()
        )

        self.actor  = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)

        # FIX 3: orthogonal init — standard PPO best practice
        # makes critic produce reasonable value estimates from episode 1
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = self.shared(state)
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, state_size, action_size):

        self.gamma        = 0.99
        self.lam          = 0.95
        self.clip         = 0.2
        self.lr           = 3e-4
        self.epochs       = 10       # FIX 4: more epochs per rollout (was 5)
        # FIX 5: higher entropy — forces exploration on sparse reward env
        # 0.01 is too low; agent collapses to deterministic policy too fast
        self.entropy_coef = 0.05
        self.value_coef   = 0.5

        self.model     = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr, eps=1e-5)
        self.reset_memory()

    def reset_memory(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def act(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_t)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value):
        values = self.values + [next_value]
        gae    = 0
        returns = []

        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + self.gamma * values[step + 1] * (1 - self.dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.lam * (1 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])

        advantages = np.array(returns) - np.array(self.values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, next_value):
        returns, advantages = self.compute_gae(next_value)

        states       = torch.FloatTensor(np.array(self.states))
        actions      = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        returns_t    = torch.FloatTensor(returns)
        advantages_t = torch.FloatTensor(advantages)

        # FIX 6: mini-batch updates within each epoch
        # full-batch updates on large rollouts destabilise training
        n = len(self.states)
        batch_size = 64

        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]

                logits, values = self.model(states[idx])
                dist       = Categorical(logits=logits)
                new_lp     = dist.log_prob(actions[idx])
                entropy    = dist.entropy().mean()

                ratio = (new_lp - old_log_probs[idx]).exp()
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio,
                                    1 - self.clip,
                                    1 + self.clip) * advantages_t[idx]

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = (returns_t[idx] - values.squeeze()).pow(2).mean()
                loss = (actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                # FIX 7: gradient clipping — missing from your version
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        self.reset_memory()
