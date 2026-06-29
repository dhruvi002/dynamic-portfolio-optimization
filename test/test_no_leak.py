"""
test/test_no_leak.py — Phase 2 leakage regression guards (I-3, I-4)
===================================================================
Two things Phase 2 must never regress on:

1. **HPO test-set leak (I-3).** The Ray Tune trial path must build its train and
   validation envs WITHOUT touching a single row in the [TEST_START, TEST_END]
   window, and its hard guard must fire if a test-window row ever sneaks in.

2. **Universe survivorship / look-ahead (I-4).** The trading universe must be the
   disclosed fixed neutral set (continuous Dow members), not the back-filled
   live DJ-30 — i.e. the mid-sample joiners/leavers must be excluded everywhere.

These tests use synthetic data only (no downloads, no torch) so they run in CI.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from config import (UNIVERSE, EXCLUDED_FROM_DJ30,
                    TRAIN_END, TEST_START, TEST_END)
from environment.portfolio_env import PortfolioEnv
from tuning import tune_runner


# ── Universe leak-free (I-4) ────────────────────────────────────────────────────

def test_universe_excludes_mid_sample_members():
    """SHW, INTC, DOW, AMGN, CRM, HON must not be in the trading universe."""
    for tic in ("SHW", "INTC", "DOW", "AMGN", "CRM", "HON"):
        assert tic not in UNIVERSE, f"{tic} leaks survivorship/look-ahead bias"
        assert tic in EXCLUDED_FROM_DJ30, f"{tic} must be documented as excluded"


def test_env_and_pipeline_use_the_same_universe():
    """The env and its DJ30 alias both resolve to the leak-free universe."""
    from data.pipeline import DJ30_TICKERS as pipeline_tickers
    assert PortfolioEnv.UNIVERSE == list(UNIVERSE)
    assert PortfolioEnv.DJ30_TICKERS == list(UNIVERSE)   # backward-compat alias
    assert pipeline_tickers == list(UNIVERSE)
    assert len(UNIVERSE) == 24


# ── HPO never reads the test window (I-3) ───────────────────────────────────────

def _force_synthetic(monkeypatch):
    """Make the tuning path fall back to synthetic full-span data (no parquet)."""
    def _raise(*a, **k):
        raise FileNotFoundError("forced synthetic for test")
    monkeypatch.setattr("pandas.read_parquet", _raise)


def test_trial_envs_contain_no_test_window_rows(monkeypatch):
    _force_synthetic(monkeypatch)
    train_env, val_env = tune_runner._build_trial_envs({})

    test_lo, test_hi = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    train_end_ts = pd.Timestamp(TRAIN_END)
    for e in (train_env, val_env):
        dates = [pd.Timestamp(d) for d in e.dates]
        assert max(dates) <= train_end_ts, "tuning env runs past TRAIN_END"
        assert not any(test_lo <= d <= test_hi for d in dates), \
            "tuning env contains test-window rows — HPO test-set leak"


def test_leak_guard_fires_if_test_rows_injected(monkeypatch):
    """If a broken split puts test-window rows in the train slot, guard must raise."""
    _force_synthetic(monkeypatch)
    full = tune_runner._synthetic_df()
    full["date"] = pd.to_datetime(full["date"])
    test_rows = full[(full["date"] >= TEST_START) & (full["date"] <= TEST_END)].copy()
    val_rows  = full[(full["date"] >= "2022-01-01") & (full["date"] <= TRAIN_END)].copy()

    # Sabotage the split: hand back test-window rows where train_df belongs.
    monkeypatch.setattr(
        "data.pipeline.three_way_split",
        lambda *a, **k: (test_rows.reset_index(drop=True),
                         val_rows.reset_index(drop=True),
                         test_rows.reset_index(drop=True)),
    )
    with pytest.raises(AssertionError, match="LEAK GUARD"):
        tune_runner._build_trial_envs({})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
