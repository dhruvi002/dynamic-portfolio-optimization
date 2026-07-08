"""
Baseline portfolio strategies for benchmark comparison.

All functions use the same cost model as PortfolioEnv:
    cost = (tc_rate + slip_rate) * sum(|Δw|) * portfolio_value

For min_variance and max_sharpe_mvo the covariance / mean-return parameters
are estimated on train_df (no look-ahead into the test period).
For momentum_12_1 weights are recomputed at each rebalance using only data
available up to that date.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy.optimize import minimize

from utils.metrics import compute_all_metrics


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pivot_prices(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Pivot long-format df (date, tic, close) to wide (date × ticker)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="tic", values="close")
    available = [t for t in tickers if t in pivot.columns]
    return pivot[available].sort_index()


def _apply_costs(
    portfolio_value: float,
    old_weights: np.ndarray,
    new_weights: np.ndarray,
    tc_rate: float,
    slip_rate: float,
) -> float:
    """Deduct rebalancing costs, matching PortfolioEnv's formula exactly."""
    turnover = np.abs(new_weights - old_weights).sum()
    return portfolio_value - (tc_rate + slip_rate) * turnover * portfolio_value


def _run_rebalancing(
    pivot: pd.DataFrame,
    target_fn,
    initial_capital: float,
    rebalance_freq: str,
    tc_rate: float,
    slip_rate: float,
) -> Tuple[dict, np.ndarray, list]:
    """
    Generic rebalancing loop.

    target_fn(date, pivot_up_to_date, current_weights) → new target weights.

    Weights drift due to price movements between rebalance dates so that
    turnover (and therefore transaction costs) are computed against the
    actual holdings just before each rebalance, not the stale target.
    """
    n = len(pivot.columns)
    pivot = pivot.dropna()
    rebal_dates = set(pivot.resample(rebalance_freq).last().index)

    weights = np.ones(n) / n
    portfolio_value = initial_capital
    values = [initial_capital]
    dates = [pivot.index[0]]

    for i in range(1, len(pivot)):
        date = pivot.index[i]
        prev_p = pivot.iloc[i - 1].values
        curr_p = pivot.iloc[i].values

        price_mult = curr_p / (prev_p + 1e-8)
        port_return = float(np.dot(weights, price_mult))
        portfolio_value *= port_return

        # Drift weights with price movements (buy-and-hold between rebalances)
        weights = weights * price_mult
        w_sum = weights.sum()
        if w_sum > 1e-8:
            weights /= w_sum

        if date in rebal_dates:
            new_w = target_fn(date, pivot.iloc[: i + 1], weights)
            portfolio_value = _apply_costs(portfolio_value, weights, new_w, tc_rate, slip_rate)
            weights = new_w

        values.append(portfolio_value)
        dates.append(date)

    values_arr = np.array(values)
    returns_arr = np.diff(values_arr) / (values_arr[:-1] + 1e-10)
    return compute_all_metrics(returns_arr, values_arr), values_arr, dates


# ── MVO helpers ───────────────────────────────────────────────────────────────

def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance portfolio weights."""
    n = cov.shape[0]
    w0 = np.ones(n) / n
    result = minimize(
        lambda w: w @ cov @ w,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if result.success:
        w = np.maximum(result.x, 0.0)
        return w / w.sum()
    return w0


def _max_sharpe_weights(mean_ret: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Long-only maximum Sharpe portfolio weights."""
    n = len(mean_ret)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        ret = w @ mean_ret
        std = np.sqrt(max(w @ cov @ w, 1e-12))
        return -ret / std

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if result.success:
        w = np.maximum(result.x, 0.0)
        s = w.sum()
        return w / s if s > 1e-8 else w0
    return w0


# ── Public baseline functions ─────────────────────────────────────────────────

def equal_weight(
    df: pd.DataFrame,
    tickers: List[str],
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    Monthly rebalance to equal weights with transaction costs.
    Returns (metrics_dict, portfolio_values, dates).
    """
    pivot = _pivot_prices(df, tickers)
    n = len(pivot.columns)
    target = np.ones(n) / n
    return _run_rebalancing(pivot, lambda d, p, w: target.copy(),
                            initial_capital, rebalance_freq, tc_rate, slip_rate)


def spy_qqq(
    start: str,
    end: str,
    weight_spy: float = 0.6,
    weight_qqq: float = 0.4,
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    60/40 SPY/QQQ blend with monthly rebalance and transaction costs.
    Downloads SPY and QQQ from yfinance for [start, end].
    Returns (metrics_dict, portfolio_values, dates).
    """
    import yfinance as yf

    raw = yf.download(["SPY", "QQQ"], start=start, end=end, auto_adjust=True,
                      progress=False, multi_level_index=True)["Close"]
    raw = raw[["SPY", "QQQ"]].dropna()
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    target = np.array([weight_spy, weight_qqq])
    return _run_rebalancing(raw, lambda d, p, w: target.copy(),
                            initial_capital, rebalance_freq, tc_rate, slip_rate)


def momentum_12_1(
    df: pd.DataFrame,
    tickers: List[str],
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    top_pct: float = 0.1,
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    Momentum 12-1: rank by 11-month return (12m ago → 1m ago, skip last month).
    Long top-decile (top_pct of tickers), equal-weighted within that set.
    Parameters computed from data available up to each rebalance date (no look-ahead).
    Returns (metrics_dict, portfolio_values, dates).
    """
    pivot = _pivot_prices(df, tickers).dropna()
    n = len(pivot.columns)
    top_n = max(1, int(round(n * top_pct)))
    monthly = pivot.resample(rebalance_freq).last()
    cols = list(pivot.columns)

    def momentum_target(date, _pivot_slice, old_weights):
        past = monthly[monthly.index <= date]
        if len(past) < 13:
            return np.ones(n) / n
        ret_12m_ago = past.iloc[-13]
        ret_1m_ago  = past.iloc[-2]
        momentum = (ret_1m_ago - ret_12m_ago) / (ret_12m_ago.abs() + 1e-8)
        ranked = momentum.rank(ascending=False)
        top_tickers = ranked[ranked <= top_n].index.tolist()
        new_w = np.zeros(n)
        for t in top_tickers:
            if t in cols:
                new_w[cols.index(t)] = 1.0 / len(top_tickers)
        return new_w if new_w.sum() > 1e-8 else old_weights.copy()

    return _run_rebalancing(pivot, momentum_target,
                            initial_capital, rebalance_freq, tc_rate, slip_rate)


def min_variance(
    df: pd.DataFrame,
    tickers: List[str],
    train_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    Minimum-variance portfolio. Covariance estimated from train_df (no look-ahead).
    Fixed optimal weights rebalanced monthly over df (test period).
    Returns (metrics_dict, portfolio_values, dates).
    """
    est_df = train_df if train_df is not None else df
    est_pivot = _pivot_prices(est_df, tickers).dropna()
    cov = est_pivot.pct_change().dropna().cov().values

    target = _min_variance_weights(cov)
    pivot = _pivot_prices(df, tickers)
    return _run_rebalancing(pivot, lambda d, p, w: target.copy(),
                            initial_capital, rebalance_freq, tc_rate, slip_rate)


def max_sharpe_mvo(
    df: pd.DataFrame,
    tickers: List[str],
    train_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    Maximum-Sharpe MVO (long-only). Parameters estimated from train_df (no look-ahead).
    Fixed optimal weights rebalanced monthly over df (test period).
    Returns (metrics_dict, portfolio_values, dates).
    """
    est_df = train_df if train_df is not None else df
    est_pivot = _pivot_prices(est_df, tickers).dropna()
    est_returns = est_pivot.pct_change().dropna()
    mean_ret = est_returns.mean().values
    cov = est_returns.cov().values

    target = _max_sharpe_weights(mean_ret, cov)
    pivot = _pivot_prices(df, tickers)
    return _run_rebalancing(pivot, lambda d, p, w: target.copy(),
                            initial_capital, rebalance_freq, tc_rate, slip_rate)


# ── Phase 4: strong baseline set (I-7) ─────────────────────────────────────────
# Adds the standard benchmarks a practitioner would actually respect: SPY
# buy-and-hold (the canonical market benchmark), a real 60/40 SPY/AGG balanced
# portfolio, inverse-volatility risk parity, and rolling Ledoit-Wolf MVO — all
# net of cost, same cost model, no look-ahead (estimators use only data with
# date <= rebalance date).

def _cached_yf_download(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """
    Memoized yfinance download keyed by (tickers, start, end). Repeated exact
    calls (e.g. re-running a harness) hit the cache instead of the network.

    NOTE: this does NOT dedupe across *different* (start, end) windows — e.g.
    walk-forward's per-fold baselines each use a different fold window, so this
    alone won't prevent per-fold downloads. Callers that iterate many
    overlapping windows (walk_forward_eval.py) should instead download the
    FULL span ONCE and pass sliced price Series into the `*_from_series`
    variants below (`spy_buy_and_hold_from_series`, `spy_agg_60_40_from_series`)
    rather than calling `spy_buy_and_hold`/`spy_agg_60_40` per fold — a Phase-4
    run hit multi-hour stalls from repeated per-fold yfinance calls before this
    split was added.
    """
    import yfinance as yf
    if len(tickers) == 1:
        raw = yf.download(tickers[0], start=start, end=end, auto_adjust=True, progress=False)["Close"]
    else:
        raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                          progress=False, multi_level_index=True)["Close"]
        raw = raw[list(tickers)]
    raw = raw.dropna()
    raw.index = pd.to_datetime(raw.index)
    return raw.sort_index()


_YF_CACHE: dict = {}


def _yf_download(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    key = (tickers, str(start), str(end))
    if key not in _YF_CACHE:
        _YF_CACHE[key] = _cached_yf_download(tickers, start, end)
    return _YF_CACHE[key]


def _values_from_prices(prices: np.ndarray, initial_capital: float, tc_rate: float, slip_rate: float) -> np.ndarray:
    """Single entry-cost buy-and-hold value path (no further rebalancing)."""
    entry_cost = (tc_rate + slip_rate) * 1.0 * initial_capital
    capital_after_entry = initial_capital - entry_cost
    shares = capital_after_entry / (prices[0] + 1e-12)
    return shares * prices


def spy_buy_and_hold_from_series(
    spy_prices: pd.Series,
    initial_capital: float = 1e6,
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    SPY buy-and-hold computed from an ALREADY-DOWNLOADED price series (sliced
    to the desired window by the caller). Use this in loops over many windows
    (e.g. walk-forward folds) to avoid a yfinance call per window — download
    the full span once with `spy_buy_and_hold`'s helper `_yf_download` (or your
    own fetch) and slice per window.
    """
    s = spy_prices.dropna().sort_index()
    prices = np.asarray(s.values, dtype=float).flatten()
    dates = list(s.index)
    values = _values_from_prices(prices, initial_capital, tc_rate, slip_rate)
    returns = np.diff(values) / (values[:-1] + 1e-10)
    return compute_all_metrics(returns, values), values, dates


def spy_buy_and_hold(
    start: str,
    end: str,
    initial_capital: float = 1e6,
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    Single-asset SPY buy-and-hold. Enters the full position once at t0 (paying
    the entry turnover cost, Σ|Δw| = 1, exactly like PortfolioEnv), then holds
    with no further rebalancing — the canonical market benchmark. Downloads SPY
    from yfinance for [start, end] (cached; see `_yf_download`).
    Returns (metrics_dict, portfolio_values, dates).
    """
    raw = _yf_download(("SPY",), start, end)
    s = pd.Series(np.asarray(raw.values, dtype=float).flatten(), index=raw.index)
    return spy_buy_and_hold_from_series(s, initial_capital, tc_rate, slip_rate)


def spy_agg_60_40_from_series(
    spy_prices: pd.Series,
    agg_prices: pd.Series,
    weight_spy: float = 0.6,
    weight_agg: float = 0.4,
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """60/40 SPY/AGG from ALREADY-DOWNLOADED price series (see `spy_buy_and_hold_from_series`)."""
    raw = pd.DataFrame({"SPY": spy_prices, "AGG": agg_prices}).dropna().sort_index()
    target = np.array([weight_spy, weight_agg])
    return _run_rebalancing(raw, lambda d, p, w: target.copy(),
                            initial_capital, rebalance_freq, tc_rate, slip_rate)


def spy_agg_60_40(
    start: str,
    end: str,
    weight_spy: float = 0.6,
    weight_agg: float = 0.4,
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    The classic 60/40 balanced benchmark: 60% SPY / 40% AGG (US aggregate
    bonds), monthly rebalance, transaction costs. This — not `spy_qqq` (two
    tech-heavy equity ETFs) — is what "60/40" means in practice. Downloads
    SPY+AGG from yfinance for [start, end] (cached; see `_yf_download`).
    Returns (metrics_dict, portfolio_values, dates).
    """
    raw = _yf_download(("SPY", "AGG"), start, end)
    return spy_agg_60_40_from_series(
        raw["SPY"], raw["AGG"], weight_spy, weight_agg,
        initial_capital, rebalance_freq, tc_rate, slip_rate,
    )


def _inverse_vol_weights(returns_window: pd.DataFrame, n: int) -> np.ndarray:
    """Inverse-volatility weights (simple risk-parity approximation)."""
    if len(returns_window) < 5:
        return np.ones(n) / n
    vol = returns_window.std().values
    inv_vol = 1.0 / (vol + 1e-8)
    w = inv_vol / inv_vol.sum()
    return w


def risk_parity(
    df: pd.DataFrame,
    tickers: List[str],
    train_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    lookback: int = 252,
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    Inverse-volatility risk-parity weights over the universe, re-estimated at
    each rebalance from a trailing `lookback`-day window of data available up
    to that date (no look-ahead). `train_df`, if given, supplies history for
    the lookback warmup before the test period starts (mirrors min_variance's
    train_df pattern) — estimation still never uses a date after the rebalance.
    Returns (metrics_dict, portfolio_values, dates).
    """
    hist_df = pd.concat([train_df, df], ignore_index=True) if train_df is not None else df
    hist_pivot = _pivot_prices(hist_df, tickers).dropna()
    pivot = _pivot_prices(df, tickers)
    n = len(pivot.columns)

    def target_fn(date, _pivot_slice, old_weights):
        past = hist_pivot[hist_pivot.index <= date]
        if len(past) < 20:
            return np.ones(n) / n
        window = past.iloc[-lookback:]
        rets = window.pct_change().dropna()
        return _inverse_vol_weights(rets, n)

    return _run_rebalancing(pivot, target_fn, initial_capital, rebalance_freq, tc_rate, slip_rate)


def _shrunk_covariance(returns_window: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance; falls back to sample covariance if
    scikit-learn is not installed (guarded, lazy import — mirrors the Ray
    fallback pattern elsewhere in this codebase)."""
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns_window.values)
        return lw.covariance_
    except ImportError:
        print("  WARNING: scikit-learn not installed; rolling_mvo_ledoit_wolf "
              "falling back to sample covariance (no shrinkage). "
              "`pip install scikit-learn` for the intended estimator.")
        return returns_window.cov().values


def rolling_mvo_ledoit_wolf(
    df: pd.DataFrame,
    tickers: List[str],
    kind: str = "min_var",
    train_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 1e6,
    rebalance_freq: str = "ME",
    lookback: int = 252,
    tc_rate: float = 0.001,
    slip_rate: float = 0.001,
) -> Tuple[dict, np.ndarray, list]:
    """
    "MVO done properly": mean-variance optimization with Ledoit-Wolf shrinkage,
    RE-ESTIMATED at each rebalance on a trailing `lookback`-day window (rolling,
    not static like `min_variance`/`max_sharpe_mvo`). Long-only, weights sum to
    1. `kind` selects `_min_variance_weights` or `_max_sharpe_weights` fed the
    shrunk covariance / rolling mean. No look-ahead: estimation at rebalance
    date t uses only data with date <= t.
    Returns (metrics_dict, portfolio_values, dates).
    """
    if kind not in ("min_var", "max_sharpe"):
        raise ValueError(f"kind must be 'min_var' or 'max_sharpe', got {kind!r}")

    hist_df = pd.concat([train_df, df], ignore_index=True) if train_df is not None else df
    hist_pivot = _pivot_prices(hist_df, tickers).dropna()
    pivot = _pivot_prices(df, tickers)
    n = len(pivot.columns)

    def target_fn(date, _pivot_slice, old_weights):
        past = hist_pivot[hist_pivot.index <= date]
        if len(past) < 20:
            return np.ones(n) / n
        window = past.iloc[-lookback:]
        rets = window.pct_change().dropna()
        if len(rets) < 5:
            return np.ones(n) / n
        cov = _shrunk_covariance(rets)
        if kind == "min_var":
            return _min_variance_weights(cov)
        mean_ret = rets.mean().values
        return _max_sharpe_weights(mean_ret, cov)

    return _run_rebalancing(pivot, target_fn, initial_capital, rebalance_freq, tc_rate, slip_rate)
