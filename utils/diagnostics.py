"""
Policy-behavior & evaluation diagnostics.
=========================================
Phase 1 (Honest Evaluation Harness). Resolves I-5, supports I-2/I-11.

Two groups of helpers:

1. **Policy behavior** — turnover, weight concentration (HHI), and active share
   vs equal-weight, computed from an environment's recorded weight history after
   a deterministic backtest. These explain *what the agent actually does* and,
   in particular, whether it simply tracks equal-weight (which would explain a
   small or insignificant edge).

2. **Aggregation** — mean ± std and a bootstrap 95% CI across seeds, so the
   headline table reports a distribution rather than a single lucky draw.

Also includes `returns_from_values`, the net-of-cost daily return series derived
from the portfolio value path — used to reconcile the total_return vs ann_return
discrepancy flagged in the Phase 0 handover (§5). See `reconcile_returns`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

TRADING_DAYS = 252


# ── Return-series reconciliation ────────────────────────────────────────────────

def returns_from_values(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Net-of-cost simple daily returns implied by the portfolio value path.

    The env's `info['port_return']` series is *gross of transaction/slippage
    costs* (costs are deducted from `portfolio_value` separately), so metrics
    built from it disagree with the actual equity curve. Returns derived here
    are consistent with `total_return` and the plotted curve.
    """
    v = np.asarray(portfolio_values, dtype=float)
    return np.diff(v) / (v[:-1] + 1e-12)


def reconcile_returns(returns_gross: np.ndarray, portfolio_values: np.ndarray) -> dict:
    """
    Explain the total_return vs ann_return sign discrepancy (handover §5).

    Two effects combine:
      (a) `returns_gross` (from info['port_return']) excludes the costs that the
          equity curve pays, so its mean is biased upward;
      (b) the *arithmetic* annualization (1+mean)^252−1 ignores volatility drag,
          whereas the geometric path (V_T/V_0)^(252/T)−1 does not.

    Returns both annualizations plus the net-of-cost series stats so the gap is
    visible and attributable rather than mysterious.
    """
    rg = np.asarray(returns_gross, dtype=float)
    v = np.asarray(portfolio_values, dtype=float)
    rn = returns_from_values(v)
    T = rn.size
    total_return = float(v[-1] / v[0] - 1.0)

    ann_arith_gross = float((1.0 + rg.mean()) ** TRADING_DAYS - 1.0)
    ann_arith_net = float((1.0 + rn.mean()) ** TRADING_DAYS - 1.0)
    ann_geom_net = float((v[-1] / v[0]) ** (TRADING_DAYS / max(T, 1)) - 1.0)

    return {
        "total_return": total_return,
        "ann_return_arith_gross": ann_arith_gross,   # the misleading legacy number
        "ann_return_arith_net": ann_arith_net,       # costs included, still arithmetic
        "ann_return_geom_net": ann_geom_net,         # the honest annualized figure
        "mean_daily_gross": float(rg.mean()),
        "mean_daily_net": float(rn.mean()),
        "cost_drag_daily": float(rg.mean() - rn.mean()),
        "vol_drag_note": "geom < arith by ~0.5·sigma^2 per period (volatility drag)",
    }


# ── Policy-behavior diagnostics ─────────────────────────────────────────────────

def turnover_series(weights_history: np.ndarray) -> np.ndarray:
    """Per-step turnover Σ|Δw_i| from a (T, N) weight history."""
    w = np.asarray(weights_history, dtype=float)
    return np.abs(np.diff(w, axis=0)).sum(axis=1)


def hhi_series(weights_history: np.ndarray) -> np.ndarray:
    """Herfindahl–Hirschman concentration index Σ w_i² per step (1/N..1)."""
    w = np.asarray(weights_history, dtype=float)
    return (w ** 2).sum(axis=1)


def active_share_series(
    weights_history: np.ndarray,
    benchmark_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Active share vs a benchmark = 0.5·Σ|w_i − b_i| per step.
    Defaults the benchmark to equal-weight (1/N). 0 = identical to benchmark,
    1 = fully disjoint holdings.
    """
    w = np.asarray(weights_history, dtype=float)
    n = w.shape[1]
    b = np.full(n, 1.0 / n) if benchmark_weights is None else np.asarray(benchmark_weights, float)
    return 0.5 * np.abs(w - b).sum(axis=1)


def policy_diagnostics(
    env,
    benchmark_weights: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute policy-behavior diagnostics from an env *after* a deterministic
    backtest has populated `env.history`.

    Returns scalar summaries plus the full per-step series (for plotting):
    turnover, HHI, active share, the mean weight vector, and a near-uniform
    flag (max |mean_w − 1/N| < 0.01) that, if True, plainly explains an
    equal-weight-like result.
    """
    weights = np.asarray(env.history["weights"], dtype=float)   # (T+1, N)
    values = np.asarray(env.history["portfolio_value"], dtype=float)
    n = weights.shape[1]

    turn = turnover_series(weights)
    hhi = hhi_series(weights)
    active = active_share_series(weights, benchmark_weights)
    mean_w = weights.mean(axis=0)
    max_dev_from_uniform = float(np.abs(mean_w - 1.0 / n).max())

    return {
        "n_steps": int(weights.shape[0] - 1),
        "n_assets": int(n),
        "mean_turnover": float(turn.mean()),
        "median_turnover": float(np.median(turn)),
        "mean_hhi": float(hhi.mean()),
        "hhi_uniform_floor": float(1.0 / n),
        "mean_active_share": float(active.mean()),
        "max_mean_weight_dev_from_uniform": max_dev_from_uniform,
        "near_uniform": bool(max_dev_from_uniform < 0.01),
        "final_value": float(values[-1]),
        # full series for plotting
        "turnover_series": turn,
        "hhi_series": hhi,
        "active_share_series": active,
        "mean_weights": mean_w,
        "portfolio_values": values,
    }


# ── Aggregation across seeds ────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple:
    """Percentile bootstrap CI on the mean of `values` (resampling with replacement)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def aggregate_metrics(
    per_seed: list,
    keys: Optional[list] = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Aggregate a list of per-seed metric dicts into {metric: stats}.

    Each stats entry has: mean, std (population), ci_low, ci_high (bootstrap
    95% CI on the mean), min, max, and n (number of finite seeds).
    """
    if not per_seed:
        return {}
    if keys is None:
        keys = sorted({k for d in per_seed for k, v in d.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed if k in d and isinstance(d[k], (int, float))
                         and not isinstance(d[k], bool)], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, seed=seed)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "ci_low": lo,
            "ci_high": hi,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(vals.size),
        }
    return out


def format_aggregate_table(agg: dict, keys: Optional[list] = None) -> str:
    """Human-readable 'metric: mean ± std  [ci_low, ci_high]  (n)' table."""
    keys = keys or list(agg.keys())
    lines = [f"  {'Metric':<18}{'Mean':>10}{'Std':>9}{'95% CI':>22}{'n':>4}", "  " + "─" * 61]
    for k in keys:
        if k not in agg:
            continue
        s = agg[k]
        ci = f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
        lines.append(f"  {k:<18}{s['mean']:>+10.4f}{s['std']:>9.4f}{ci:>22}{s['n']:>4}")
    return "\n".join(lines)
