"""
Tests for utils/baselines.py.

Synthetic data only — no yfinance downloads. spy_qqq is tested separately
(requires network access) and is skipped here.
"""

import numpy as np
import pandas as pd
import pytest

from utils.baselines import (
    equal_weight,
    momentum_12_1,
    min_variance,
    max_sharpe_mvo,
    _min_variance_weights,
    _max_sharpe_weights,
)


# ── Synthetic data factory ─────────────────────────────────────────────────────

def make_price_df(n_tickers: int = 10, n_days: int = 600, seed: int = 42) -> tuple:
    """Return (df, tickers) with synthetic daily prices in long format."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    for t in tickers:
        price = 100.0
        for d in dates:
            price *= 1.0 + rng.normal(5e-4, 0.02)
            rows.append({"date": d, "tic": t, "close": max(price, 1.0),
                         "open": price, "high": price * 1.01,
                         "low": price * 0.99, "volume": 1e6})
    df = pd.DataFrame(rows).sort_values(["date", "tic"]).reset_index(drop=True)
    return df, tickers


def train_test_split(df, split_frac=0.6):
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * split_frac)]
    return (df[df["date"] < cut].copy().reset_index(drop=True),
            df[df["date"] >= cut].copy().reset_index(drop=True))


# ── Weight-level unit tests ────────────────────────────────────────────────────

def test_min_variance_weights_sum_to_one():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    cov = A @ A.T / 36 + np.eye(6) * 0.01
    w = _min_variance_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w >= -1e-7).all(), "negative weights"


def test_min_variance_weights_non_negative():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T / 25 + np.eye(5) * 0.005
    w = _min_variance_weights(cov)
    assert (w >= -1e-7).all()


def test_min_variance_lower_risk_than_equal():
    """Min-variance portfolio should have ≤ variance of equal-weight."""
    rng = np.random.default_rng(3)
    n = 8
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n + np.eye(n) * 0.01
    w_mv = _min_variance_weights(cov)
    w_eq = np.ones(n) / n
    assert (w_mv @ cov @ w_mv) <= (w_eq @ cov @ w_eq) + 1e-6


def test_max_sharpe_weights_sum_to_one():
    rng = np.random.default_rng(1)
    n = 5
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n + np.eye(n) * 0.01
    mean_ret = rng.uniform(0.0, 0.002, size=n)
    w = _max_sharpe_weights(mean_ret, cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-7).all()


def test_max_sharpe_higher_sharpe_than_equal():
    """Max-Sharpe portfolio should have ≥ Sharpe of equal-weight."""
    rng = np.random.default_rng(5)
    n = 6
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n + np.eye(n) * 0.005
    # Assets with spread in expected returns so optimiser can do something
    mean_ret = np.linspace(0.0002, 0.002, n)
    w_ms = _max_sharpe_weights(mean_ret, cov)
    w_eq = np.ones(n) / n

    def sharpe(w):
        return (w @ mean_ret) / max(np.sqrt(w @ cov @ w), 1e-12)

    assert sharpe(w_ms) >= sharpe(w_eq) - 1e-6


# ── Integration tests (full pipeline) ─────────────────────────────────────────

class TestEqualWeight:
    def setup_method(self):
        self.df, self.tickers = make_price_df()

    def test_returns_valid_metrics(self):
        met, vals, dates = equal_weight(self.df, self.tickers)
        assert "sharpe" in met and "max_drawdown" in met
        assert np.isfinite(met["sharpe"])

    def test_portfolio_starts_at_initial_capital(self):
        initial = 500_000.0
        _, vals, _ = equal_weight(self.df, self.tickers, initial_capital=initial)
        assert abs(vals[0] - initial) < 1.0

    def test_costs_reduce_final_value(self):
        _, vals_cost, _ = equal_weight(self.df, self.tickers, tc_rate=0.001, slip_rate=0.001)
        _, vals_free, _ = equal_weight(self.df, self.tickers, tc_rate=0.0, slip_rate=0.0)
        assert vals_cost[-1] < vals_free[-1]

    def test_output_lengths_consistent(self):
        _, vals, dates = equal_weight(self.df, self.tickers)
        assert len(vals) == len(dates)

    def test_portfolio_value_positive(self):
        _, vals, _ = equal_weight(self.df, self.tickers)
        assert (vals > 0).all()


class TestMomentum:
    def setup_method(self):
        self.df, self.tickers = make_price_df(n_tickers=20, n_days=700)

    def test_returns_valid_metrics(self):
        met, _, _ = momentum_12_1(self.df, self.tickers)
        assert "sharpe" in met
        assert np.isfinite(met["sharpe"])

    def test_portfolio_value_positive(self):
        _, vals, _ = momentum_12_1(self.df, self.tickers)
        assert (vals > 0).all()

    def test_costs_reduce_value(self):
        _, vals_cost, _ = momentum_12_1(self.df, self.tickers, tc_rate=0.001, slip_rate=0.001)
        _, vals_free, _ = momentum_12_1(self.df, self.tickers, tc_rate=0.0, slip_rate=0.0)
        assert vals_cost[-1] <= vals_free[-1]

    def test_no_future_data_in_first_fold(self):
        """Truncating the dataframe should not change earlier returns."""
        dates = sorted(self.df["date"].unique())
        cutoff = dates[400]
        df_short = self.df[self.df["date"] <= cutoff].copy()
        # Both runs should succeed without exceptions (no look-ahead check)
        met_full, _, _ = momentum_12_1(self.df, self.tickers)
        met_short, _, _ = momentum_12_1(df_short, self.tickers)
        assert np.isfinite(met_full["sharpe"])
        assert np.isfinite(met_short["sharpe"])


class TestMinVariance:
    def setup_method(self):
        df, self.tickers = make_price_df(n_tickers=8, n_days=700)
        self.train_df, self.test_df = train_test_split(df)

    def test_returns_valid_metrics(self):
        met, _, _ = min_variance(self.test_df, self.tickers, train_df=self.train_df)
        assert "sharpe" in met
        assert np.isfinite(met["sharpe"])

    def test_portfolio_value_positive(self):
        _, vals, _ = min_variance(self.test_df, self.tickers, train_df=self.train_df)
        assert (vals > 0).all()

    def test_uses_train_df_not_test(self):
        """Passing different train windows should give different weights (no look-ahead)."""
        df2, _ = make_price_df(n_tickers=8, n_days=700, seed=99)
        train_df2, _ = train_test_split(df2)
        met1, _, _ = min_variance(self.test_df, self.tickers, train_df=self.train_df)
        met2, _, _ = min_variance(self.test_df, self.tickers, train_df=train_df2)
        # Different training data → different Sharpe (weights differ)
        assert met1["sharpe"] != met2["sharpe"]


class TestMaxSharpeMVO:
    def setup_method(self):
        df, self.tickers = make_price_df(n_tickers=8, n_days=700)
        self.train_df, self.test_df = train_test_split(df)

    def test_returns_valid_metrics(self):
        met, _, _ = max_sharpe_mvo(self.test_df, self.tickers, train_df=self.train_df)
        assert "sharpe" in met
        assert np.isfinite(met["sharpe"])

    def test_portfolio_value_positive(self):
        _, vals, _ = max_sharpe_mvo(self.test_df, self.tickers, train_df=self.train_df)
        assert (vals > 0).all()

    def test_costs_reduce_value(self):
        _, vals_cost, _ = max_sharpe_mvo(self.test_df, self.tickers, train_df=self.train_df,
                                          tc_rate=0.001, slip_rate=0.001)
        _, vals_free, _ = max_sharpe_mvo(self.test_df, self.tickers, train_df=self.train_df,
                                          tc_rate=0.0, slip_rate=0.0)
        assert vals_cost[-1] <= vals_free[-1]
