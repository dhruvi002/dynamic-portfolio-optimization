"""
Soft Actor-Critic (SAC) Agent for Portfolio Optimization
=========================================================
Implements SAC with:
  - Dirichlet policy for portfolio weights on the K-simplex
  - Automatic entropy tuning with Dirichlet-correct target entropy
  - Twin critics to reduce Q-value overestimation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Standard replay buffer."""

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


class AssetTransformerEncoder(nn.Module):
    """
    Treats each asset as a sequence token with its per-asset features.

    State is feature-major: [feat0×n_assets | feat1×n_assets | ...].
    We reshape to [B, n_features, n_assets] then transpose to [B, n_assets, n_features]
    so each asset attends to all others with full cross-asset attention.
    """

    def __init__(self, n_features: int = 6, n_assets: int = 30,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_features = n_features
        self.n_assets   = n_assets
        self.embed       = nn.Linear(n_features, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.out_dim     = n_assets * d_model

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        # state is feature-major → [B, n_features, n_assets] → [B, n_assets, n_features]
        x = state.view(B, self.n_features, self.n_assets).transpose(1, 2)
        x = self.embed(x)         # [B, n_assets, d_model]
        x = self.transformer(x)   # [B, n_assets, d_model]
        return x.flatten(1)       # [B, n_assets * d_model]


class DirichletActor(nn.Module):
    """
    Dirichlet policy for portfolio weights on the K-simplex.

    Outputs concentration parameters α_i ∈ [CONC_MIN, CONC_MAX] via
    softplus + clamp, then samples via PyTorch's reparameterized Dirichlet
    (implicit reparameterization through Gamma).  log_prob() is the exact
    Dirichlet log-density — no Jacobian approximations needed.
    """
    CONC_MIN = 1e-2
    CONC_MAX = 50.0

    def __init__(self, state_dim: int, action_dim: int, hidden: list = None,
                 encoder: str = "mlp", n_assets: int = 30):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        self.use_transformer = encoder == "transformer"
        if self.use_transformer:
            n_features       = state_dim // n_assets
            self.encoder     = AssetTransformerEncoder(n_features=n_features, n_assets=n_assets)
            net_in_dim       = self.encoder.out_dim
        else:
            net_in_dim = state_dim
        self.net       = _mlp(net_in_dim, hidden[:-1], hidden[-1])
        self.conc_head = nn.Linear(hidden[-1], action_dim)

    def _concentrations(self, state: torch.Tensor) -> torch.Tensor:
        h    = self.encoder(state) if self.use_transformer else state
        x    = F.relu(self.net(h))
        conc = F.softplus(self.conc_head(x)) + 1e-3
        return conc.clamp(self.CONC_MIN, self.CONC_MAX)

    def sample(self, state: torch.Tensor):
        """
        Returns (action, log_prob, mean).
          action   : [B, K]  — reparameterized sample on simplex
          log_prob : [B, 1]  — Dirichlet log-density at action
          mean     : [B, K]  — distribution mean (α_i / Σ α_j)
        """
        conc     = self._concentrations(state)
        dist     = torch.distributions.Dirichlet(conc)
        action   = dist.rsample()                          # implicit reparam
        log_prob = dist.log_prob(action).unsqueeze(1)      # [B, 1]
        return action, log_prob, dist.mean

    def mean_action(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic action: Dirichlet mean = α_i / Σ α_j."""
        conc = self._concentrations(state)
        return torch.distributions.Dirichlet(conc).mean


# Keep old name as alias so existing imports don't break.
Actor = DirichletActor


class Critic(nn.Module):
    """Twin Q-networks (Q1, Q2) — takes (state, action) → Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden: list = None,
                 encoder: str = "mlp", n_assets: int = 30):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        self.use_transformer = encoder == "transformer"
        if self.use_transformer:
            n_features   = state_dim // n_assets
            self.encoder = AssetTransformerEncoder(n_features=n_features, n_assets=n_assets)
            in_dim       = self.encoder.out_dim + action_dim
        else:
            in_dim = state_dim + action_dim
        self.q1 = _mlp(in_dim, hidden, 1)
        self.q2 = _mlp(in_dim, hidden, 1)

    def _encode(self, state, action):
        enc = self.encoder(state) if self.use_transformer else state
        return torch.cat([enc, action], dim=-1)

    def forward(self, state, action):
        x = self._encode(state, action)
        return self.q1(x), self.q2(x)

    def q1_value(self, state, action):
        x = self._encode(state, action)
        return self.q1(x)


# ─── SAC Agent ────────────────────────────────────────────────────────────────

def dirichlet_symmetric_entropy(action_dim: int, conc: float) -> float:
    """
    Differential entropy of a symmetric Dirichlet(conc · 1_K) on the K-simplex.

    Phase 5 (Task A, I-6): used to derive a target entropy the clamped
    `DirichletActor` can actually attain. The Dirichlet differential entropy is
    maximised at the uniform policy (conc=1, H = -lgamma(K)) and falls off on
    BOTH sides — collapsing to -inf as conc→0 and decreasing again as conc grows.
    The old target `-(lgamma(K) + 0.5·K)` sat far below the uniform maximum
    (for K=24: -63.6 vs -51.6, ~conc 5.5), demanding a concentration the actor
    only reaches under strong critic pressure; with no clamp on log-alpha the
    auto-tuner ran away (α → 9.6 blow-up) or, when the critic over-concentrated,
    collapsed (α → 1e-3). Targeting a *mild* symmetric concentration keeps the
    policy near-diversified (which also curbs turnover — see Phase 5 Task B)
    while remaining an attainable, interpretable fixed point.

        H = K·lgamma(c) − lgamma(K·c) + (K·c − K)·ψ(K·c) − K·(c − 1)·ψ(c)

    with c = conc, ψ = digamma.
    """
    c = float(conc)
    K = int(action_dim)
    a0 = K * c
    t = torch.tensor([c, a0], dtype=torch.float64)
    lgamma_c, lgamma_a0 = torch.lgamma(t).tolist()
    digamma_c, digamma_a0 = torch.digamma(t).tolist()
    return (K * lgamma_c - lgamma_a0
            + (a0 - K) * digamma_a0
            - K * (c - 1.0) * digamma_c)


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
        encoder: str = "mlp",
        target_entropy: float = None,
        target_conc: float = 2.0,
        alpha_init: float = 1.0,
        alpha_min: float = 0.01,
        alpha_max: float = 5.0,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        # n_assets == action_dim for this project (one weight per asset)
        n_assets = action_dim

        # Networks
        self.actor        = DirichletActor(state_dim, action_dim, hidden_sizes, encoder=encoder, n_assets=n_assets).to(self.device)
        self.critic       = Critic(state_dim, action_dim, hidden_sizes, encoder=encoder, n_assets=n_assets).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_sizes, encoder=encoder, n_assets=n_assets).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Automatic entropy tuning with a Dirichlet-attainable target and a
        # bounded temperature (Phase 5, Task A, I-6).
        #
        # Differential entropy of a Dirichlet is MAXIMISED at the uniform policy
        # (H_uniform = -lgamma(K); -51.6 nats for K=24) and decreases on both
        # sides. The old target -(lgamma(K)+0.5K) = -63.6 sat well below that
        # maximum (~symmetric conc 5.5), so the tuner chased a concentration the
        # actor rarely reaches and — with no clamp — log-alpha ran away (α→9.6)
        # or collapsed (α→1e-3) across seeds/folds. We instead target the
        # entropy of a MILD symmetric Dirichlet(target_conc·1_K) (default
        # conc=2.0 → -54.3 nats, just below uniform): an attainable fixed point
        # that keeps the policy near-diversified, which also curbs turnover.
        if target_entropy is None:
            target_entropy = dirichlet_symmetric_entropy(action_dim, target_conc)
        self.target_entropy = float(target_entropy)
        self.target_conc    = float(target_conc)

        # Bound the temperature so neither blow-up nor collapse can recur; the
        # log_alpha tensor is clamped in-place after every update() step.
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self._log_alpha_min = math.log(self.alpha_min)
        self._log_alpha_max = math.log(self.alpha_max)
        init_log_alpha = min(max(math.log(alpha_init), self._log_alpha_min),
                             self._log_alpha_max)
        self.log_alpha = torch.tensor(init_log_alpha, requires_grad=True,
                                      device=self.device)
        self.alpha = self.log_alpha.exp().item()

        # Optimisers
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_opt  = optim.Adam([self.log_alpha],          lr=lr_alpha)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.total_steps   = 0

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            # Use the Dirichlet mean — same distribution as training, no double transform.
            action = self.actor.mean_action(state_t)
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
            target_q   = rewards + self.gamma * (1 - dones) * min_q_next

        q1, q2      = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        # ── Actor loss ────────────────────────────────────────────────────────
        new_actions, log_pi, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        min_q_new      = torch.min(q1_new, q2_new)
        actor_loss     = (self.alpha * log_pi - min_q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        # ── Entropy temperature (alpha) loss ──────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # Bound the temperature (Phase 5, Task A) so it can neither blow up nor
        # collapse regardless of transient target/critic pressure.
        with torch.no_grad():
            self.log_alpha.clamp_(self._log_alpha_min, self._log_alpha_max)
        self.alpha = self.log_alpha.exp().item()

        # ── Soft-update target critic ─────────────────────────────────────────
        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.copy_(self.tau * p.data + (1 - self.tau) * p_tgt.data)

        self.total_steps += 1

        return {
            "critic_loss":   critic_loss.item(),
            "actor_loss":    actor_loss.item(),
            "alpha_loss":    alpha_loss.item(),
            "alpha":         self.alpha,
            # Mean policy entropy estimate (−E[log π]); paired with alpha this
            # quantifies the entropy collapse diagnosed in Phase 1 (I-6).
            "policy_entropy": float((-log_pi).mean().item()),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha,
            "total_steps":   self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"]
        self.alpha     = self.log_alpha.exp().item()
        self.total_steps = ckpt["total_steps"]
