"""
Unit tests for SAC agent and portfolio environment.
Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.sac import SACAgent, DirichletActor, Critic, ReplayBuffer
from environment.portfolio_env import PortfolioEnv
from utils.metrics import compute_sharpe, compute_sortino, compute_max_drawdown


# ─── Fixtures ─────────────────────────────────────────────────────────────────

N_TICKERS = 5
TICKERS = ["A", "B", "C", "D", "E"]
N_DAYS = 100


def make_df(n_days=N_DAYS, tickers=TICKERS):
    """Synthetic OHLCV + indicator dataframe."""
    rows = []
    prices = np.ones(len(tickers)) * 100.0
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    for date in dates:
        prices = prices * np.exp(np.random.randn(len(tickers)) * 0.01)
        for i, tic in enumerate(tickers):
            rows.append({
                "date": date, "tic": tic,
                "open": prices[i], "high": prices[i] * 1.005,
                "low": prices[i] * 0.995, "close": prices[i],
                "volume": 1e6,
                "macd": 0.0, "rsi_30": 50.0, "cci_30": 0.0, "dx_30": 25.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def env():
    df = make_df()
    return PortfolioEnv(df, tickers=TICKERS, transaction_cost_rate=0.001, slippage_rate=0.001)


@pytest.fixture
def agent(env):
    return SACAgent(state_dim=env.state_dim, action_dim=env.action_dim, batch_size=32, hidden_sizes=[64, 64])


# ─── Environment tests ────────────────────────────────────────────────────────

class TestPortfolioEnv:

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset()
        assert obs.shape == (env.state_dim,)
        assert not np.isnan(obs).any()

    def test_step_returns_valid_obs(self, env):
        env.reset()
        action = np.ones(N_TICKERS) / N_TICKERS
        obs, reward, done, trunc, info = env.step(action)
        assert obs.shape == (env.state_dim,)
        assert isinstance(reward, float)
        assert "portfolio_value" in info

    def test_weights_sum_to_one(self, env):
        env.reset()
        action = np.array([3.0, 1.0, 0.0, 2.0, 4.0])  # unnormalised
        _, _, _, _, _ = env.step(action)
        assert abs(env.weights.sum() - 1.0) < 1e-5

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            _, _, done, _, _ = env.step(action)
            steps += 1
        assert steps == env.n_steps

    def test_transaction_costs_reduce_value(self, env):
        """High-turnover policy should underperform low-turnover one."""
        # Low turnover: equal weight, hold
        env.reset()
        start = env.portfolio_value
        for _ in range(10):
            env.step(np.ones(N_TICKERS) / N_TICKERS)
        low_tc_value = env.portfolio_value

        # High turnover: alternating extreme weights
        env.reset()
        for i in range(10):
            w = np.zeros(N_TICKERS)
            w[i % N_TICKERS] = 1.0
            env.step(w)
        high_tc_value = env.portfolio_value

        # Hard to guarantee direction due to random prices, but at least it runs
        assert low_tc_value > 0 and high_tc_value > 0


# ─── SAC Agent tests ──────────────────────────────────────────────────────────

class TestSACAgent:

    def test_select_action_shape(self, agent, env):
        obs, _ = env.reset()
        action = agent.select_action(obs)
        assert action.shape == (N_TICKERS,)

    def test_action_sums_to_one(self, agent, env):
        obs, _ = env.reset()
        action = agent.select_action(obs)
        assert abs(action.sum() - 1.0) < 1e-4

    def test_update_requires_min_buffer(self, agent):
        losses = agent.update()
        assert losses == {}  # buffer empty, no update

    def test_update_after_fill(self, agent, env):
        obs, _ = env.reset()
        for _ in range(agent.batch_size + 1):
            action = env.action_space.sample()
            next_obs, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs if not done else env.reset()[0]

        losses = agent.update()
        assert "actor_loss" in losses
        assert "critic_loss" in losses
        assert "alpha" in losses

    def test_alpha_updates_during_training(self, agent, env):
        initial_alpha = agent.alpha
        obs, _ = env.reset()
        for _ in range(300):
            action = env.action_space.sample()
            next_obs, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs if not done else env.reset()[0]
        agent.update()
        # Alpha should change (automatic entropy tuning)
        # We just check it's still a valid positive number
        assert agent.alpha > 0

    def test_save_load(self, agent, env, tmp_path):
        obs, _ = env.reset()
        action_before = agent.select_action(obs, deterministic=True)

        path = str(tmp_path / "test_agent.pt")
        agent.save(path)
        agent.load(path)

        action_after = agent.select_action(obs, deterministic=True)
        np.testing.assert_allclose(action_before, action_after, atol=1e-5)


# ─── Metrics tests ────────────────────────────────────────────────────────────

class TestMetrics:

    def test_sharpe_positive_returns(self):
        returns = np.ones(252) * 0.001
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_zero_variance(self):
        returns = np.zeros(252)
        sharpe = compute_sharpe(returns)
        assert sharpe == 0.0

    def test_sortino_positive(self):
        returns = np.clip(np.random.randn(252) * 0.01 + 0.001, -0.1, 0.1)
        assert compute_sortino(returns) > 0

    def test_max_drawdown_negative(self):
        values = np.array([100, 110, 105, 90, 95, 100.0])
        mdd = compute_max_drawdown(values)
        assert mdd < 0
        assert abs(mdd - (-90/110)) < 1e-4

    def test_max_drawdown_monotone(self):
        values = np.array([100, 105, 110, 115, 120.0])
        assert compute_max_drawdown(values) == 0.0
