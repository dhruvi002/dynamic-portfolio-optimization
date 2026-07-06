"""
test/test_seeding.py — Phase 5 Task C: reproducibility / determinism guards.

Covers the two nondeterminism sources fixed in Phase 5:
  1. Uncontrolled CPU threading (OpenMP/MKL float reduction order).
  2. env.action_space.sample() drawing from an unseeded RNG during warm-up,
     which randomised the whole replay buffer across identical-seed runs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils.seeding import set_global_seed
from environment.portfolio_env import PortfolioEnv


# ── Thread pinning ─────────────────────────────────────────────────────────────

def test_thread_env_vars_pinned_on_import():
    # utils.seeding sets these at import time so the numerical backends spin up
    # single-threaded (fixed float summation order → determinism on CPU).
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ.get(var) == "1", f"{var} not pinned to 1"


def test_set_global_seed_echoes_seed():
    assert set_global_seed(123) == 123


def test_torch_single_threaded_after_seed():
    torch = pytest.importorskip("torch")
    set_global_seed(7)
    assert torch.get_num_threads() == 1


# ── Action-space RNG (the primary §12 leak) ────────────────────────────────────

def _make_df(n_tickers=5, n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows, prices = [], np.ones(n_tickers) * 100.0
    import pandas as pd
    for d in pd.date_range("2020-01-01", periods=n_days, freq="B"):
        prices = prices * np.exp(rng.standard_normal(n_tickers) * 0.01)
        for i, t in enumerate(tickers):
            rows.append({"date": d, "tic": t, "open": prices[i],
                         "high": prices[i] * 1.01, "low": prices[i] * 0.99,
                         "close": prices[i], "volume": 1e6,
                         "macd": 0.0, "rsi_30": 50.0, "cci_30": 0.0, "dx_30": 25.0})
    return pd.DataFrame(rows), tickers


def test_action_space_seeding_is_reproducible():
    # Two identically-seeded envs must yield identical action_space.sample()
    # sequences. Without env.action_space.seed(seed) (Phase 5, trainer.train)
    # these draw from a randomly-initialised RNG and diverge.
    df, tickers = _make_df()

    def sample_seq(seed):
        env = PortfolioEnv(df, tickers=tickers)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return [env.action_space.sample() for _ in range(10)]

    a = sample_seq(42)
    b = sample_seq(42)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_action_space_different_seeds_differ():
    df, tickers = _make_df()

    def first_sample(seed):
        env = PortfolioEnv(df, tickers=tickers)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env.action_space.sample()

    assert not np.array_equal(first_sample(1), first_sample(2))
