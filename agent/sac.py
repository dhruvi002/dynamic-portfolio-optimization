"""
Soft Actor-Critic (SAC) Agent for Portfolio Optimization
=========================================================
Implements SAC with:
  - Continuous action space (portfolio weights ∈ [0,1], sum=1)
  - Automatic entropy tuning (target_entropy = -dim(A))
  - Twin critics to reduce Q-value overestimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque
import random


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Standard replay buffer. Swap for PrioritizedReplayBuffer for PER."""

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)).unsqueeze(1),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Networks ─────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: list, out_dim: int, activation=nn.ReLU):
    layers = []
    dims = [in_dim] + hidden
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Gaussian policy. Outputs mean & log_std, then samples via reparameterisation.
    Applies softmax so actions represent portfolio weights summing to 1.
    """
    LOG_STD_MIN, LOG_STD_MAX = -5, 2

    def __init__(self, state_dim: int, action_dim: int, hidden: list = [256, 256]):
        super().__init__()
        self.net = _mlp(state_dim, hidden[:-1], hidden[-1])
        self.mean_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)

    def forward(self, state):
        x = F.relu(self.net(state))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()                          # reparameterisation trick
        action_raw = torch.tanh(z)                  # squash to (-1, 1)
        action = F.softmax(action_raw, dim=-1)      # portfolio weights

        # Log prob with tanh squashing correction
        log_prob = dist.log_prob(z) - torch.log(1 - action_raw.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """Twin Q-networks (Q1, Q2) — takes (state, action) → Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden: list = [256, 256]):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = _mlp(in_dim, hidden, 1)
        self.q2 = _mlp(in_dim, hidden, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


# ─── SAC Agent ────────────────────────────────────────────────────────────────

class SACAgent:
    """
    Soft Actor-Critic agent with automatic entropy temperature tuning.

    Key hyperparameters (all tunable via Ray Tune):
        gamma         : discount factor
        tau           : soft update rate for target networks
        lr_actor      : actor learning rate
        lr_critic     : critic learning rate
        lr_alpha      : entropy temperature learning rate
        batch_size    : mini-batch size
        hidden_sizes  : list of hidden layer widths
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        hidden_sizes: list = None,
        device: str = "auto",
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Automatic entropy tuning
        self.target_entropy = -action_dim  # heuristic: -|A|
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()

        # Optimisers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.total_steps = 0

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            _, _, action = self.actor.sample(state_t)
            action = F.softmax(action, dim=-1)
        else:
            action, _, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy()

    # ── Learning step ─────────────────────────────────────────────────────────

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = [
            t.to(self.device) for t in self.replay_buffer.sample(self.batch_size)
        ]

        # ── Critic loss ───────────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards + self.gamma * (1 - dones) * min_q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        # ── Actor loss ────────────────────────────────────────────────────────
        new_actions, log_pi, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - min_q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        # ── Entropy temperature (alpha) loss ──────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # ── Soft-update target critic ─────────────────────────────────────────
        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.copy_(self.tau * p.data + (1 - self.tau) * p_tgt.data)

        self.total_steps += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha,
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"]
        self.alpha = self.log_alpha.exp().item()
        self.total_steps = ckpt["total_steps"]
