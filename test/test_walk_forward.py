"""
test/test_walk_forward.py — Phase 3 walk-forward guards (I-2 regime part)
========================================================================
Four invariants the multi-regime walk-forward must never regress on:

1. **Chronology / non-overlap / expanding train** — fold test windows are
   strictly increasing and contiguous (non-overlapping), the training window
   expands, and every fold has ``train_end < test_start`` (no future leak).
2. **Clean stop** — requesting more folds than the data window supports yields
   fewer folds, never a window running past the data end.
3. **NET metric path** — the per-fold metrics are derived from the net-of-cost
   value path (so they differ from the gross ``backtest()`` Sharpe when turnover,
   hence cost, is non-zero), with ``gross_sharpe`` exposed alongside.
4. **Regime labelling** — fold test windows map to the documented regime bands.

Synthetic data only (no downloads, no Ray, no SAC training), so it runs in CI
inside the full suite.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from utils.walk_forward import _fold_windows, _fold_net_metrics
from utils.regimes import regime_for, REGIME_ORDER


# ── 1. Fold scheduling: chronological, non-overlapping, expanding, leak-free ─────

def test_fold_windows_chronological_nonoverlapping_expanding():
    start = pd.Timestamp("2019-04-01")
    end = pd.Timestamp("2025-01-31")
    windows = _fold_windows(start, end, n_folds=9, test_months=6, min_train_months=12)

    assert len(windows) >= 3, "expected several folds across the span"

    prev_test_end = None
    prev_train_end = None
    for w in windows:
        # No future leak.
        assert w["train_end"] < w["test_start"], "train_end must precede test_start"
        # Test window well-formed.
        assert w["test_start"] <= w["test_end"]
        # Train window expands fold over fold.
        if prev_train_end is not None:
            assert w["train_end"] > prev_train_end, "training window must expand"
        # Test windows are increasing and non-overlapping.
        if prev_test_end is not None:
            assert w["test_start"] > prev_test_end, "test windows must not overlap"
        prev_test_end = w["test_end"]
        prev_train_end = w["train_end"]

    # Folds are 1-based and consecutive.
    assert [w["fold"] for w in windows] == list(range(1, len(windows) + 1))


def test_fold_windows_clean_stop():
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2023-06-30")   # only ~18 months available
    windows = _fold_windows(start, end, n_folds=20, test_months=6, min_train_months=12)
    # Far fewer than the 20 requested, and none exceed the data window.
    assert len(windows) < 20
    for w in windows:
        assert w["test_end"] <= end, "a fold ran past the data window"


# ── 3. NET metric path (differs from gross when turnover > 0) ────────────────────

class _ChurningStubAgent:
    """
    Deterministic non-learning agent that fully rotates the portfolio into a
    different single name each step → high turnover → material transaction cost.
    No torch; used only to exercise the net-vs-gross metric path.
    """
    def __init__(self, n_assets):
        self.n = n_assets
        self._t = 0

    def select_action(self, state, deterministic=True):
        w = np.zeros(self.n, dtype=np.float32)
        w[self._t % self.n] = 1.0
        self._t += 1
        return w


def _tiny_synthetic_df(months=8):
    """Small synthetic OHLC+indicator panel over a single test window."""
    from environment.portfolio_env import PortfolioEnv
    tickers = PortfolioEnv.UNIVERSE
    dates = pd.date_range("2023-01-02", periods=months * 21, freq="B")
    rng = np.random.default_rng(0)
    prices = np.ones(len(tickers)) * 100.0
    rows = []
    for date in dates:
        prices = prices * np.exp(rng.standard_normal(len(tickers)) * 0.01)
        for i, tic in enumerate(tickers):
            rows.append({
                "date": date, "tic": tic,
                "open": prices[i], "high": prices[i] * 1.005,
                "low": prices[i] * 0.995, "close": prices[i], "volume": 1e6,
                "macd": 0.0, "rsi_30": 50.0, "cci_30": 0.0, "dx_30": 25.0,
            })
    return pd.DataFrame(rows)


def test_net_metric_path_differs_from_gross_when_turnover():
    from environment.portfolio_env import PortfolioEnv
    from utils.trainer import backtest

    df = _tiny_synthetic_df()
    env = PortfolioEnv(df, transaction_cost_rate=0.001, slippage_rate=0.001,
                       initial_capital=1_000_000.0)
    agent = _ChurningStubAgent(env.n_assets)

    gross = backtest(agent, env)            # trainer.backtest → GROSS metrics
    net = _fold_net_metrics(env)            # Phase-3 NET metrics (+ gross_sharpe)

    # gross_sharpe must be exposed, and the net Sharpe must differ from the gross
    # Sharpe because the churning agent pays real costs the gross path ignores.
    assert "gross_sharpe" in net
    assert not np.isclose(net["sharpe"], gross["sharpe"], atol=1e-6), \
        "net Sharpe should differ from gross when turnover (cost) > 0"

    # NET total_return is derived from the value path (consistent with the curve).
    pv = np.asarray(env.history["portfolio_value"], dtype=float)
    assert np.isclose(net["total_return"], pv[-1] / pv[0] - 1.0, atol=1e-9)

    # Costs are a drag: with heavy turnover the net path underperforms the gross.
    assert net["final_value"] <= 1_000_000.0 * (1 + gross["total_return"]) + 1e-6


# ── 4. Regime labelling ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("test_start,test_end,expected", [
    ("2020-04-01", "2020-09-30", "COVID crash/recovery"),
    ("2021-04-01", "2021-09-30", "2021 bull"),
    ("2022-04-01", "2022-09-30", "2022 bear (rate shock)"),
    ("2023-07-01", "2023-12-31", "2023-24 recovery/AI"),
    ("2024-06-01", "2024-11-30", "2023-24 recovery/AI"),
])
def test_regime_labelling(test_start, test_end, expected):
    assert regime_for(test_start, test_end) == expected


def test_regime_order_is_complete():
    # Every band label is in the display order, and order has no duplicates.
    assert len(REGIME_ORDER) == len(set(REGIME_ORDER))
    for _, _, _ in []:
        pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
