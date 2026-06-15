"""
test/test_env.py - Portfolio environment unit tests
Guards Bug 1: prev_value must be captured before portfolio_value is overwritten.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from environment.portfolio_env import PortfolioEnv


# ── Synthetic data factory (seeded, no downloads) ─────────────────────────────

def make_df(n_tickers=5, n_days=80, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    prices = np.ones(n_tickers) * 100.0
    for d in pd.date_range("2020-01-01", periods=n_days, freq="B"):
        prices = prices * np.exp(rng.standard_normal(n_tickers) * 0.01)
        for i, t in enumerate(tickers):
            rows.append({"date": d, "tic": t, "open": prices[i],
                         "high": prices[i]*1.01, "low": prices[i]*0.99,
                         "close": prices[i], "volume": 1e6,
                         "macd": 0.0, "rsi_30": 50.0, "cci_30": 0.0, "dx_30": 25.0})
    return pd.DataFrame(rows), tickers


@pytest.fixture
def env():
    df, tickers = make_df()
    return PortfolioEnv(df, tickers=tickers, transaction_cost_rate=0.001, slippage_rate=0.001)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_reward_nonzero(env):
    # Guards Bug 1: with old code log(net/net)≈0 every step; fixed code gives log(net/prev)≠0.
    obs, _ = env.reset()
    action = np.ones(env.n_assets) / env.n_assets
    nonzero, steps = 0, 0
    done = False
    while not done:
        _, reward, done, _, _ = env.step(action)
        if abs(reward) > 1e-10:
            nonzero += 1
        steps += 1
    assert nonzero / steps >= 0.8, f"Only {nonzero}/{steps} steps had non-zero reward"


def test_weights_sum_to_one(env):
    env.reset()
    unnormalized = np.array([3.0, 1.0, 0.0, 2.0, 4.0], dtype=np.float32)
    env.step(unnormalized)
    assert abs(env.weights.sum() - 1.0) < 1e-5


def test_episode_length(env):
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, done, _, _ = env.step(np.ones(env.n_assets) / env.n_assets)
        steps += 1
    assert steps == env.n_steps


def test_step_increments(env):
    env.reset()
    for expected in range(1, 6):
        env.step(np.ones(env.n_assets) / env.n_assets)
        assert env.current_step == expected


def test_done_flag(env):
    env.reset()
    action = np.ones(env.n_assets) / env.n_assets
    for _ in range(env.n_steps - 1):
        _, _, done, _, _ = env.step(action)
        assert not done
    _, _, done, _, _ = env.step(action)
    assert done


def test_portfolio_value_tracked(env):
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, done, _, _ = env.step(np.ones(env.n_assets) / env.n_assets)
        steps += 1
    assert len(env.history["portfolio_value"]) == steps + 1


def test_obs_shape(env):
    obs, _ = env.reset()
    assert obs.shape == (env.state_dim,)
    assert not np.isnan(obs).any()
    obs, _, _, _, _ = env.step(np.ones(env.n_assets) / env.n_assets)
    assert obs.shape == (env.state_dim,)
    assert not np.isnan(obs).any()


def test_transaction_costs_applied(env):
    env.reset()
    n = env.n_assets
    # First step establishes non-equal weights, second step flips to opposite end → high turnover
    w1 = np.zeros(n, dtype=np.float32); w1[0] = 1.0
    w2 = np.zeros(n, dtype=np.float32); w2[-1] = 1.0
    env.step(w1)
    _, _, _, _, info = env.step(w2)
    assert info["tc"] > 0


# ── Sentiment integration ─────────────────────────────────────────────────────

def make_sentiment_df(tickers, dates, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "tic": t,
                         "sentiment_score": float(rng.uniform(-1, 1))})
    return pd.DataFrame(rows)


def test_sentiment_state_dim():
    df, tickers = make_df()
    dates = sorted(df["date"].unique())
    sentiment_df = make_sentiment_df(tickers, dates)
    env_plain = PortfolioEnv(df, tickers=tickers)
    env_sent  = PortfolioEnv(df, tickers=tickers, sentiment_df=sentiment_df)
    assert env_sent.state_dim == env_plain.state_dim + len(tickers), \
        "sentiment should add n_assets dims to state_dim"


def test_sentiment_obs_shape():
    df, tickers = make_df()
    dates = sorted(df["date"].unique())
    sentiment_df = make_sentiment_df(tickers, dates)
    env = PortfolioEnv(df, tickers=tickers, sentiment_df=sentiment_df)
    obs, _ = env.reset()
    assert obs.shape == (env.state_dim,)
    assert not np.isnan(obs).any()


def test_sentiment_missing_dates_fill_zero():
    # Provide sentiment for only half the dates — missing ones should get 0.0
    df, tickers = make_df()
    dates = sorted(df["date"].unique())
    # Only supply sentiment for even-indexed dates
    partial_dates = [dates[i] for i in range(0, len(dates), 2)]
    sentiment_df = make_sentiment_df(tickers, partial_dates)
    env = PortfolioEnv(df, tickers=tickers, sentiment_df=sentiment_df)
    # Step through entire episode; state should never be NaN
    env.reset()
    done = False
    while not done:
        obs, _, done, _, _ = env.step(np.ones(env.n_assets) / env.n_assets)
        assert not np.isnan(obs).any(), "NaN found in obs with partial sentiment coverage"


def test_no_sentiment_unchanged_state_dim():
    df, tickers = make_df()
    env = PortfolioEnv(df, tickers=tickers)
    n = len(tickers)
    expected = n + n + n * 4   # weights + returns + 4 tech indicators
    assert env.state_dim == expected
