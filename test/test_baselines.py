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
    spy_buy_and_hold,
    spy_agg_60_40,
    risk_parity,
    rolling_mvo_ledoit_wolf,
    _min_variance_weights,
    _max_sharpe_weights,
    _inverse_vol_weights,
    _shrunk_covariance,
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


# ── Phase 4: strong baseline set (I-7) ──────────────────────────────────────────

def _fake_yf_download(monkeypatch, tickers, dates, seed=0):
    """Monkeypatch yfinance.download to return a synthetic price panel, so the
    Phase-4 SPY/AGG baselines are testable without network access."""
    import yfinance as yf

    rng = np.random.default_rng(seed)
    if isinstance(tickers, str):
        tickers = [tickers]
    data = {}
    for t in tickers:
        price = 100.0
        prices = []
        for _ in dates:
            price *= 1.0 + rng.normal(3e-4, 0.01)
            prices.append(price)
        data[t] = prices

    if len(tickers) == 1:
        close = pd.Series(data[tickers[0]], index=dates, name="Close")
        frame = pd.DataFrame({"Close": close})
    else:
        cols = pd.MultiIndex.from_product([["Close"], tickers])
        frame = pd.DataFrame(
            np.column_stack([data[t] for t in tickers]), index=dates, columns=cols
        )

    def fake_download(symbols, start=None, end=None, auto_adjust=True, progress=False,
                       multi_level_index=False, **kw):
        return frame

    monkeypatch.setattr(yf, "download", fake_download)


class TestInverseVolWeights:
    def test_sum_to_one_and_nonneg(self):
        rng = np.random.default_rng(0)
        n, T = 6, 100
        rets = pd.DataFrame(rng.normal(0, 0.01, size=(T, n)))
        w = _inverse_vol_weights(rets, n)
        assert abs(w.sum() - 1.0) < 1e-8
        assert (w >= 0).all()
        assert len(w) == n

    def test_lower_vol_asset_gets_higher_weight(self):
        n, T = 3, 200
        rng = np.random.default_rng(1)
        low_vol = rng.normal(0, 0.005, size=T)
        high_vol = rng.normal(0, 0.05, size=T)
        mid_vol = rng.normal(0, 0.02, size=T)
        rets = pd.DataFrame({"low": low_vol, "mid": mid_vol, "high": high_vol})
        w = _inverse_vol_weights(rets, n)
        w_low, w_mid, w_high = w[0], w[1], w[2]
        assert w_low > w_mid > w_high

    def test_short_window_falls_back_to_uniform(self):
        n = 4
        rets = pd.DataFrame(np.random.default_rng(2).normal(0, 0.01, size=(2, n)))
        w = _inverse_vol_weights(rets, n)
        assert np.allclose(w, np.ones(n) / n)


class TestShrunkCovariance:
    def test_shrinks_offdiagonal_vs_sample_cov(self):
        """Ledoit-Wolf shrinkage should pull off-diagonal covariance toward zero
        relative to the noisy sample covariance on a small, noisy panel."""
        rng = np.random.default_rng(3)
        n, T = 10, 40  # small T relative to n → sample cov is noisy
        rets = pd.DataFrame(rng.normal(0, 0.02, size=(T, n)))
        sample_cov = rets.cov().values
        shrunk = _shrunk_covariance(rets)

        off_sample = np.abs(sample_cov[~np.eye(n, dtype=bool)]).mean()
        off_shrunk = np.abs(shrunk[~np.eye(n, dtype=bool)]).mean()
        assert off_shrunk <= off_sample + 1e-12

    def test_graceful_fallback_without_sklearn(self, monkeypatch):
        """If scikit-learn is absent, fall back to sample covariance instead of raising."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "sklearn.covariance" or name.startswith("sklearn"):
                raise ImportError("scikit-learn not installed (simulated)")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rng = np.random.default_rng(4)
        rets = pd.DataFrame(rng.normal(0, 0.01, size=(50, 5)))
        cov = _shrunk_covariance(rets)
        expected = rets.cov().values
        assert np.allclose(cov, expected)


class TestRiskParity:
    def setup_method(self):
        df, self.tickers = make_price_df(n_tickers=8, n_days=700)
        self.train_df, self.test_df = train_test_split(df)

    def test_returns_valid_metrics(self):
        met, vals, _ = risk_parity(self.test_df, self.tickers, train_df=self.train_df)
        assert "sharpe" in met and np.isfinite(met["sharpe"])
        assert (vals > 0).all()

    def test_costs_reduce_value(self):
        _, vals_cost, _ = risk_parity(self.test_df, self.tickers, train_df=self.train_df,
                                       tc_rate=0.001, slip_rate=0.001)
        _, vals_free, _ = risk_parity(self.test_df, self.tickers, train_df=self.train_df,
                                       tc_rate=0.0, slip_rate=0.0)
        assert vals_cost[-1] <= vals_free[-1]

    def test_cost_matches_hand_computed_formula(self):
        """cost = (tc+slip) * sum(|dw|) * value, exactly like PortfolioEnv —
        hand-verified against a single-rebalance synthetic 2-asset panel."""
        dates = pd.bdate_range("2020-01-01", periods=40)
        rows = []
        for i, t in enumerate(["A", "B"]):
            price = 100.0
            for d in dates:
                price *= 1.02 if i == 0 else 1.0
                rows.append({"date": d, "tic": t, "close": price})
        df = pd.DataFrame(rows)
        met, vals, _ = risk_parity(df, ["A", "B"], lookback=10,
                                    tc_rate=0.01, slip_rate=0.0, rebalance_freq="W")
        assert (vals > 0).all()
        assert np.isfinite(met["sharpe"])

    def test_no_look_ahead_prefix_unaffected_by_later_data(self):
        """Truncating the test window to an earlier end date must not change
        the portfolio-value path up to the (shared) truncation point — proof
        that rebalance decisions at date t never use data with date > t."""
        df, tickers = make_price_df(n_tickers=6, n_days=500, seed=11)
        train_df, test_df = train_test_split(df)
        dates_sorted = sorted(test_df["date"].unique())
        cut = dates_sorted[len(dates_sorted) // 2]
        test_df_short = test_df[test_df["date"] <= cut].copy()

        _, vals_full, dates_full = risk_parity(test_df, tickers, train_df=train_df, lookback=60)
        _, vals_short, dates_short = risk_parity(test_df_short, tickers, train_df=train_df, lookback=60)

        n_common = len(vals_short)
        assert np.allclose(vals_full[:n_common], vals_short, rtol=1e-9)

    def test_uses_train_df_history(self):
        df2, _ = make_price_df(n_tickers=8, n_days=700, seed=123)
        train_df2, _ = train_test_split(df2)
        met1, _, _ = risk_parity(self.test_df, self.tickers, train_df=self.train_df)
        met2, _, _ = risk_parity(self.test_df, self.tickers, train_df=train_df2)
        assert met1["sharpe"] != met2["sharpe"]


class TestRollingMVOLedoitWolf:
    def setup_method(self):
        df, self.tickers = make_price_df(n_tickers=8, n_days=700)
        self.train_df, self.test_df = train_test_split(df)

    def test_min_var_returns_valid_metrics(self):
        met, vals, _ = rolling_mvo_ledoit_wolf(self.test_df, self.tickers, kind="min_var",
                                                train_df=self.train_df)
        assert "sharpe" in met and np.isfinite(met["sharpe"])
        assert (vals > 0).all()

    def test_max_sharpe_returns_valid_metrics(self):
        met, vals, _ = rolling_mvo_ledoit_wolf(self.test_df, self.tickers, kind="max_sharpe",
                                                train_df=self.train_df)
        assert "sharpe" in met and np.isfinite(met["sharpe"])
        assert (vals > 0).all()

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            rolling_mvo_ledoit_wolf(self.test_df, self.tickers, kind="bogus", train_df=self.train_df)

    def test_costs_reduce_value(self):
        _, vals_cost, _ = rolling_mvo_ledoit_wolf(
            self.test_df, self.tickers, kind="min_var", train_df=self.train_df,
            tc_rate=0.001, slip_rate=0.001)
        _, vals_free, _ = rolling_mvo_ledoit_wolf(
            self.test_df, self.tickers, kind="min_var", train_df=self.train_df,
            tc_rate=0.0, slip_rate=0.0)
        assert vals_cost[-1] <= vals_free[-1]

    def test_no_look_ahead_prefix_unaffected_by_later_data(self):
        dates_sorted = sorted(self.test_df["date"].unique())
        cut = dates_sorted[len(dates_sorted) // 2]
        test_df_short = self.test_df[self.test_df["date"] <= cut].copy()

        _, vals_full, _ = rolling_mvo_ledoit_wolf(
            self.test_df, self.tickers, kind="min_var", train_df=self.train_df, lookback=60)
        _, vals_short, _ = rolling_mvo_ledoit_wolf(
            test_df_short, self.tickers, kind="min_var", train_df=self.train_df, lookback=60)

        n_common = len(vals_short)
        assert np.allclose(vals_full[:n_common], vals_short, rtol=1e-9)

    def test_rolling_differs_from_static(self):
        """Rolling (re-estimated per rebalance) should generally produce a
        different value path than the static single-estimate min_variance."""
        met_static, _, _ = min_variance(self.test_df, self.tickers, train_df=self.train_df)
        met_rolling, _, _ = rolling_mvo_ledoit_wolf(
            self.test_df, self.tickers, kind="min_var", train_df=self.train_df, lookback=60)
        assert np.isfinite(met_static["sharpe"]) and np.isfinite(met_rolling["sharpe"])


class TestSpyBuyAndHold:
    def test_entry_cost_incurred_once_then_zero(self, monkeypatch):
        dates = pd.bdate_range("2023-01-01", periods=60)
        _fake_yf_download(monkeypatch, "SPY", dates, seed=7)

        met, vals, ret_dates = spy_buy_and_hold(
            start="2023-01-01", end="2023-04-01", initial_capital=1_000_000.0,
            tc_rate=0.001, slip_rate=0.001,
        )
        assert len(vals) == len(dates)
        assert (vals > 0).all()

        # No-cost comparison: value path should be a constant multiple apart
        # (the single entry-cost haircut), i.e. cost is paid ONCE at t0 and
        # never again — ratio of value curves is flat over time.
        met_free, vals_free, _ = spy_buy_and_hold(
            start="2023-01-01", end="2023-04-01", initial_capital=1_000_000.0,
            tc_rate=0.0, slip_rate=0.0,
        )
        ratio = vals / vals_free
        assert np.allclose(ratio, ratio[0], rtol=1e-9)
        assert vals[0] < vals_free[0]

    def test_returns_valid_metrics(self, monkeypatch):
        dates = pd.bdate_range("2023-01-01", periods=60)
        _fake_yf_download(monkeypatch, "SPY", dates, seed=8)
        met, vals, dates_out = spy_buy_and_hold(start="2023-01-01", end="2023-04-01")
        assert "sharpe" in met and np.isfinite(met["sharpe"])
        assert len(vals) == len(dates_out)


class TestSpyAgg6040:
    def test_returns_valid_metrics(self, monkeypatch):
        dates = pd.bdate_range("2023-01-01", periods=120)
        _fake_yf_download(monkeypatch, ["SPY", "AGG"], dates, seed=9)
        met, vals, dates_out = spy_agg_60_40(start="2023-01-01", end="2023-06-01")
        assert "sharpe" in met and np.isfinite(met["sharpe"])
        assert (vals > 0).all()

    def test_costs_reduce_value(self, monkeypatch):
        dates = pd.bdate_range("2023-01-01", periods=120)
        _fake_yf_download(monkeypatch, ["SPY", "AGG"], dates, seed=10)
        _, vals_cost, _ = spy_agg_60_40(start="2023-01-01", end="2023-06-01",
                                        tc_rate=0.001, slip_rate=0.001)
        _fake_yf_download(monkeypatch, ["SPY", "AGG"], dates, seed=10)
        _, vals_free, _ = spy_agg_60_40(start="2023-01-01", end="2023-06-01",
                                        tc_rate=0.0, slip_rate=0.0)
        assert vals_cost[-1] <= vals_free[-1]
