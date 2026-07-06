"""
test/test_sac.py - SAC agent unit tests
Guards Bug 2 (DirichletActor output), Bug 3 (mean_action deterministic path),
and the Dirichlet target entropy fix.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tempfile
import numpy as np
import torch
import pytest

from agent.sac import (
    SACAgent, DirichletActor, AssetTransformerEncoder,
    dirichlet_symmetric_entropy,
)


STATE_DIM  = 30
ACTION_DIM = 5
BATCH      = 8


@pytest.fixture
def agent():
    return SACAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        batch_size=20, hidden_sizes=[64, 64],
    )


@pytest.fixture
def actor():
    return DirichletActor(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=[64, 64])


@pytest.fixture
def state_t():
    return torch.rand(BATCH, STATE_DIM)


@pytest.fixture
def obs():
    rng = np.random.default_rng(42)
    return rng.random(STATE_DIM).astype(np.float32)


def _fill_buffer(agent, n=20):
    rng = np.random.default_rng(0)
    for _ in range(n):
        s  = rng.random(STATE_DIM).astype(np.float32)
        a  = rng.dirichlet(np.ones(ACTION_DIM)).astype(np.float32)
        r  = float(rng.standard_normal())
        ns = rng.random(STATE_DIM).astype(np.float32)
        agent.replay_buffer.push(s, a, r, ns, 0.0)


# ── Simplex validity ──────────────────────────────────────────────────────────

def test_action_on_simplex_stochastic(agent, obs):
    # Guards Bug 2: DirichletActor must produce valid simplex weights (stochastic).
    action = agent.select_action(obs, deterministic=False)
    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= 0), "action contains negative weights"
    assert abs(action.sum() - 1.0) < 1e-5, f"action sums to {action.sum()}"


def test_action_on_simplex_deterministic(agent, obs):
    # Guards Bug 3: deterministic path must also lie on the simplex.
    action = agent.select_action(obs, deterministic=True)
    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= 0), "deterministic action contains negative weights"
    assert abs(action.sum() - 1.0) < 1e-5, f"deterministic action sums to {action.sum()}"


def test_deterministic_equals_mean_action(agent, obs):
    # Guards Bug 3: select_action(deterministic=True) must equal actor.mean_action() directly.
    # Old code applied softmax(tanh(mean)) instead of the true Dirichlet mean.
    agent.actor.eval()
    state_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        mean_direct = agent.actor.mean_action(state_t).squeeze(0).cpu().numpy()
    action = agent.select_action(obs, deterministic=True)
    np.testing.assert_allclose(action, mean_direct, atol=1e-5)


# ── Shape checks ──────────────────────────────────────────────────────────────

def test_log_prob_shape(actor, state_t):
    _, log_prob, _ = actor.sample(state_t)
    assert log_prob.shape == (BATCH, 1), f"log_prob.shape={log_prob.shape}"


def test_critic_shapes(agent, state_t):
    action_t = torch.rand(BATCH, ACTION_DIM)
    action_t = action_t / action_t.sum(dim=-1, keepdim=True)
    q1, q2 = agent.critic(state_t.to(agent.device), action_t.to(agent.device))
    assert q1.shape == (BATCH, 1)
    assert q2.shape == (BATCH, 1)


# ── Update checks ─────────────────────────────────────────────────────────────

def test_update_no_nan(agent):
    # Guards Bug 2: broken Gaussian actor produced NaN gradients; DirichletActor must not.
    _fill_buffer(agent)
    losses = agent.update()
    assert losses, "update() returned empty dict"
    for k, v in losses.items():
        assert not math.isnan(v), f"NaN in {k}"


def test_update_returns_keys(agent):
    _fill_buffer(agent)
    losses = agent.update()
    for key in ("actor_loss", "critic_loss", "alpha_loss", "alpha"):
        assert key in losses, f"missing key: {key}"


def test_alpha_positive(agent):
    _fill_buffer(agent)
    agent.update()
    assert agent.alpha > 0


# ── Entropy target ────────────────────────────────────────────────────────────

def test_target_entropy_below_hmax():
    # Guards Dirichlet target entropy fix: target must lie below H_max = -lgamma(K).
    # Old heuristic -K sat above H_max for K≥3, collapsing α to 0.
    for K in (5, 30):
        a = SACAgent(state_dim=10, action_dim=K, batch_size=4, hidden_sizes=[32])
        h_max = -math.lgamma(K)
        assert a.target_entropy < h_max, (
            f"K={K}: target_entropy={a.target_entropy:.4f} ≥ H_max={h_max:.4f}"
        )


# ── Phase 5 Task A: attainable target entropy + bounded temperature ────────────

def test_dirichlet_symmetric_entropy_matches_closed_form():
    # Cross-check against the digamma closed form for a symmetric Dirichlet.
    # K=24, conc=2.0 → -54.2916 (verified against scipy in the Phase 5 plan).
    assert dirichlet_symmetric_entropy(24, 2.0) == pytest.approx(-54.2916, abs=1e-3)
    # conc=1 is the uniform policy → H = -lgamma(K), the entropy MAXIMUM.
    assert dirichlet_symmetric_entropy(24, 1.0) == pytest.approx(-math.lgamma(24), abs=1e-4)


def test_target_entropy_defaults_to_mild_concentration():
    # Default target = entropy of a mildly-concentrated symmetric Dirichlet
    # (conc=2.0). It must sit just below the uniform maximum, not far below it
    # like the old -(lgamma(K)+0.5K) target that drove the α blow-up.
    K = 24
    a = SACAgent(state_dim=10, action_dim=K, batch_size=4, hidden_sizes=[32])
    h_uniform = -math.lgamma(K)                     # -51.61
    old_target = -(math.lgamma(K) + K * 0.5)        # -63.61
    assert a.target_entropy == pytest.approx(dirichlet_symmetric_entropy(K, 2.0), abs=1e-3)
    assert old_target < a.target_entropy < h_uniform


def test_target_entropy_configurable():
    # target_conc controls the target; an explicit target_entropy overrides it.
    a1 = SACAgent(state_dim=10, action_dim=8, batch_size=4, hidden_sizes=[32],
                  target_conc=1.5)
    assert a1.target_entropy == pytest.approx(dirichlet_symmetric_entropy(8, 1.5), abs=1e-3)
    a2 = SACAgent(state_dim=10, action_dim=8, batch_size=4, hidden_sizes=[32],
                  target_entropy=-12.34)
    assert a2.target_entropy == pytest.approx(-12.34)


def test_alpha_init_clamped_into_band():
    a = SACAgent(state_dim=10, action_dim=5, batch_size=4, hidden_sizes=[32],
                 alpha_init=100.0, alpha_min=0.01, alpha_max=5.0)
    assert a.alpha <= 5.0 + 1e-6
    assert a.alpha >= 0.01 - 1e-6


def test_alpha_cannot_blow_up():
    # The core I-6 fix: even under many updates the temperature must stay within
    # [alpha_min, alpha_max]. A tight ceiling makes the guard unambiguous.
    a = SACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, batch_size=20,
                 hidden_sizes=[64, 64], alpha_min=0.05, alpha_max=2.0,
                 lr_alpha=1.0)   # huge α LR to try to force a blow-up
    _fill_buffer(a, n=40)
    for _ in range(50):
        a.update()
        assert 0.05 - 1e-6 <= a.alpha <= 2.0 + 1e-6, f"alpha escaped band: {a.alpha}"
        assert a._log_alpha_min <= a.log_alpha.item() <= a._log_alpha_max


# ── Persistence ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip(agent, obs):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    agent.save(path)
    action_before = agent.select_action(obs, deterministic=True)

    agent2 = SACAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        batch_size=20, hidden_sizes=[64, 64],
    )
    agent2.load(path)
    action_after = agent2.select_action(obs, deterministic=True)
    np.testing.assert_allclose(action_before, action_after, atol=1e-5)


# ── Concentration clamping ────────────────────────────────────────────────────

def test_concentration_clamped(actor, state_t):
    conc = actor._concentrations(state_t)
    assert conc.min().item() >= DirichletActor.CONC_MIN, "concentration below CONC_MIN"
    assert conc.max().item() <= DirichletActor.CONC_MAX, "concentration above CONC_MAX"


# ── Transformer encoder ───────────────────────────────────────────────────────

_N_ASSETS   = 30
_D_MODEL    = 64

@pytest.mark.parametrize("n_features", [6, 7])
def test_transformer_encoder_output_shape(n_features):
    encoder = AssetTransformerEncoder(n_features=n_features, n_assets=_N_ASSETS, d_model=_D_MODEL)
    state = torch.rand(BATCH, n_features * _N_ASSETS)
    out = encoder(state)
    assert out.shape == (BATCH, _N_ASSETS * _D_MODEL), \
        f"n_features={n_features}: expected ({BATCH}, {_N_ASSETS * _D_MODEL}), got {out.shape}"


def test_transformer_actor_on_simplex():
    state_dim = 6 * _N_ASSETS   # 180
    actor_t = DirichletActor(state_dim, _N_ASSETS, hidden=[64, 64],
                              encoder="transformer", n_assets=_N_ASSETS)
    state = torch.rand(BATCH, state_dim)
    action, log_prob, mean = actor_t.sample(state)
    assert action.shape == (BATCH, _N_ASSETS)
    assert torch.all(action >= 0), "transformer actor output has negative weights"
    assert torch.allclose(action.sum(dim=-1), torch.ones(BATCH), atol=1e-5), \
        "transformer actor output does not sum to 1"


def test_transformer_critic_output_shape():
    from agent.sac import Critic
    state_dim  = 6 * _N_ASSETS
    action_dim = _N_ASSETS
    critic_t   = Critic(state_dim, action_dim, hidden=[64, 64],
                        encoder="transformer", n_assets=_N_ASSETS)
    state  = torch.rand(BATCH, state_dim)
    action = torch.rand(BATCH, action_dim)
    action = action / action.sum(dim=-1, keepdim=True)
    q1, q2 = critic_t(state, action)
    assert q1.shape == (BATCH, 1)
    assert q2.shape == (BATCH, 1)


def test_sac_agent_transformer_no_nan():
    agent_t = SACAgent(
        state_dim=6 * _N_ASSETS, action_dim=_N_ASSETS,
        batch_size=20, hidden_sizes=[64, 64], encoder="transformer",
    )
    rng = np.random.default_rng(7)
    for _ in range(20):
        s  = rng.random(6 * _N_ASSETS).astype(np.float32)
        a  = rng.dirichlet(np.ones(_N_ASSETS)).astype(np.float32)
        ns = rng.random(6 * _N_ASSETS).astype(np.float32)
        agent_t.replay_buffer.push(s, a, float(rng.standard_normal()), ns, 0.0)
    losses = agent_t.update()
    assert losses, "transformer agent update() returned empty dict"
    for k, v in losses.items():
        assert not math.isnan(v), f"NaN in {k} (transformer agent)"
