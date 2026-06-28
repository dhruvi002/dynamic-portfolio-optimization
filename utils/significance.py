"""
Statistical significance & overfitting diagnostics for Sharpe ratios.
=====================================================================
Phase 1 (Honest Evaluation Harness). Resolves the core of I-2 and I-11.

This module answers a single question rigorously: *is the agent's Sharpe ratio
actually different from the benchmark's, or is the gap noise?*

It provides:
  - `jobson_korkie_memmel`  — parametric test for the difference of two Sharpe
    ratios on correlated return series (the agent and equal-weight are both
    long-DJ30, so they are highly correlated; the Memmel (2003) correction
    accounts for that).
  - `sharpe_diff_bootstrap_ci` — a stationary (Politis–Romano) block bootstrap
    CI on the Sharpe difference, as a non-parametric cross-check that does not
    assume i.i.d. normal returns.
  - `deflated_sharpe_ratio` / `probabilistic_sharpe_ratio` — Bailey & López de
    Prado overfitting-aware Sharpe diagnostics that haircut the observed Sharpe
    for the number of trials, the sample length, and return non-normality.
  - `probability_of_backtest_overfitting` — CSCV-based PBO (optional; needs a
    matrix of per-trial return series).

Convention
----------
All Sharpe ratios used *inside the test statistics* are **per-period**
(non-annualized): ``mean(returns) / std(returns)``. The asymptotic variance
formulas are derived in per-period units; annualizing both numerator and the
sqrt of the variance cancels, so the z-statistic and p-value are identical
either way. Helper `annualize_sharpe` is provided only for human-readable
reporting.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats

TRADING_DAYS = 252


# ── Basic Sharpe helpers ────────────────────────────────────────────────────────

def periodic_sharpe(returns: np.ndarray, ddof: int = 0) -> float:
    """Per-period (non-annualized) Sharpe = mean / std."""
    returns = np.asarray(returns, dtype=float)
    sd = returns.std(ddof=ddof)
    if sd < 1e-12:
        return 0.0
    return float(returns.mean() / sd)


def annualize_sharpe(sr_periodic: float, periods_per_year: int = TRADING_DAYS) -> float:
    """Convert a per-period Sharpe to an annualized one (×sqrt(periods))."""
    return float(sr_periodic * math.sqrt(periods_per_year))


# ── Jobson–Korkie test with the Memmel correction ───────────────────────────────

def jobson_korkie_memmel(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    annualize: bool = True,
) -> dict:
    """
    Test H0: Sharpe(a) == Sharpe(b) for two *correlated* return series.

    Implements the Jobson–Korkie (1981) statistic with the Memmel (2003)
    variance correction. The two series must be aligned and equal length
    (paired observations, e.g. agent vs equal-weight on the same test dates).

    Returns a dict with per-period and (optionally) annualized Sharpe ratios,
    the Sharpe difference, the correlation, the z-statistic, and the two-sided
    p-value.

    Reference: Memmel, C. (2003), "Performance Hypothesis Testing with the
    Sharpe and Treynor Ratios", Finance Letters 1, 21–23.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"return series must align: {a.shape} vs {b.shape}")
    T = a.size
    if T < 3:
        raise ValueError("need at least 3 paired observations")

    mu_a, mu_b = a.mean(), b.mean()
    sd_a, sd_b = a.std(ddof=0), b.std(ddof=0)
    if sd_a < 1e-12 or sd_b < 1e-12:
        raise ValueError("a return series has ~zero variance; Sharpe undefined")

    sr_a = mu_a / sd_a            # per-period Sharpe ratios
    sr_b = mu_b / sd_b
    rho = float(np.corrcoef(a, b)[0, 1])
    diff = sr_a - sr_b

    # Memmel-corrected asymptotic variance of the Sharpe difference.
    theta = (
        2.0 * (1.0 - rho)
        + 0.5 * (sr_a ** 2 + sr_b ** 2 - 2.0 * sr_a * sr_b * rho ** 2)
    )
    var_diff = theta / T
    se = math.sqrt(max(var_diff, 1e-18))
    z = diff / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    out = {
        "test": "Jobson-Korkie (Memmel correction)",
        "n_obs": int(T),
        "sharpe_a_periodic": float(sr_a),
        "sharpe_b_periodic": float(sr_b),
        "sharpe_diff_periodic": float(diff),
        "correlation": rho,
        "z_stat": float(z),
        "p_value": float(p_value),
        "significant_5pct": bool(p_value < 0.05),
    }
    if annualize:
        out["sharpe_a_annual"] = annualize_sharpe(sr_a)
        out["sharpe_b_annual"] = annualize_sharpe(sr_b)
        out["sharpe_diff_annual"] = annualize_sharpe(diff)
    return out


# ── Stationary (Politis–Romano) block bootstrap CI on the Sharpe difference ─────

def _stationary_bootstrap_indices(
    n: int, avg_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Index sequence of length n from the stationary bootstrap (geometric blocks)."""
    p = 1.0 / max(avg_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)          # start a new block
        else:
            idx[t] = (idx[t - 1] + 1) % n        # continue (circular)
    return idx


def sharpe_diff_bootstrap_ci(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_boot: int = 10_000,
    avg_block: float = 10.0,
    alpha: float = 0.05,
    annualize: bool = True,
    seed: int = 0,
) -> dict:
    """
    Stationary block-bootstrap CI on the Sharpe difference (a − b).

    Resamples *paired* blocks (same indices for both series) so the
    cross-correlation between agent and benchmark is preserved. Returns the
    point estimate, the (1−alpha) percentile CI, and a bootstrap p-value
    (share of resamples with the opposite sign, doubled).

    `avg_block` is the expected block length in periods (~2–4 weeks of daily
    data is a common choice for daily financial returns).
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"return series must align: {a.shape} vs {b.shape}")
    n = a.size
    rng = np.random.default_rng(seed)

    point = periodic_sharpe(a) - periodic_sharpe(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stationary_bootstrap_indices(n, avg_block, rng)
        diffs[i] = periodic_sharpe(a[idx]) - periodic_sharpe(b[idx])

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap p-value: how often the resampled diff crosses zero.
    frac_le0 = float(np.mean(diffs <= 0.0))
    p_boot = 2.0 * min(frac_le0, 1.0 - frac_le0)

    scale = math.sqrt(TRADING_DAYS) if annualize else 1.0
    return {
        "method": "stationary block bootstrap (Politis-Romano)",
        "n_boot": int(n_boot),
        "avg_block": float(avg_block),
        "alpha": float(alpha),
        "sharpe_diff_periodic": float(point),
        "ci_periodic": (lo, hi),
        "sharpe_diff_annual": float(point * scale) if annualize else None,
        "ci_annual": (lo * scale, hi * scale) if annualize else None,
        "p_value_bootstrap": float(p_boot),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


# ── Probabilistic & Deflated Sharpe Ratio (Bailey & López de Prado) ──────────────

def probabilistic_sharpe_ratio(
    sr_periodic: float,
    n_obs: int,
    skew: float,
    kurt: float,
    sr_benchmark: float = 0.0,
) -> float:
    """
    PSR = P(true Sharpe > benchmark Sharpe), accounting for sample length and
    the non-normality (skew, kurtosis) of returns. All Sharpe inputs are
    *per-period*. `kurt` is the non-excess (raw) kurtosis (normal = 3).
    """
    denom = math.sqrt(max(1.0 - skew * sr_periodic + (kurt - 1.0) / 4.0 * sr_periodic ** 2, 1e-12))
    z = (sr_periodic - sr_benchmark) * math.sqrt(max(n_obs - 1, 1)) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_trials_std: float) -> float:
    """
    Expected maximum *per-period* Sharpe across `n_trials` independent trials
    whose Sharpe ratios have standard deviation `sr_trials_std`
    (Bailey–López de Prado, the benchmark used to deflate).
    """
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler–Mascheroni
    e = math.e
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(sr_trials_std * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    sr_periodic: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    sr_trials_std: Optional[float] = None,
) -> dict:
    """
    Deflated Sharpe Ratio (DSR): the PSR evaluated against the *expected
    maximum* Sharpe one would obtain from `n_trials` trials by luck alone.

    If `sr_trials_std` (the std of Sharpe across the trials) is unknown, pass
    the per-seed Sharpe std from the multi-seed batch as a proxy; if None, falls
    back to the variance implied by sampling noise (1/sqrt(n_obs)).

    Returns a dict with the deflation benchmark and the DSR probability. A DSR
    near 1 means the result survives the multiple-testing haircut; near 0 means
    it is plausibly an artifact of trying many configurations.
    """
    if sr_trials_std is None:
        sr_trials_std = 1.0 / math.sqrt(max(n_obs, 2))  # null sampling-noise proxy
    sr0 = expected_max_sharpe(n_trials, sr_trials_std)
    dsr = probabilistic_sharpe_ratio(sr_periodic, n_obs, skew, kurt, sr_benchmark=sr0)
    return {
        "metric": "Deflated Sharpe Ratio",
        "sr_periodic": float(sr_periodic),
        "n_trials": int(n_trials),
        "sr_trials_std": float(sr_trials_std),
        "expected_max_sharpe_periodic": float(sr0),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "dsr": float(dsr),
    }


# ── Probability of Backtest Overfitting (CSCV) — optional ───────────────────────

def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 16,
) -> dict:
    """
    PBO via Combinatorially Symmetric Cross-Validation (Bailey et al. 2017).

    `returns_matrix` is shape (T, N): T time periods × N candidate strategies
    (e.g. one column per HPO config or per seed). The function splits time into
    `n_splits` contiguous blocks, forms all balanced in-sample/out-of-sample
    partitions, selects the best strategy in-sample, and measures how often it
    ranks below median out-of-sample. PBO is the fraction of partitions where
    the IS-best strategy under-performs OOS.

    Returns {'pbo': float, 'n_combinations': int}. Requires N ≥ 2 strategies.
    """
    from itertools import combinations

    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("returns_matrix must be (T, N) with N >= 2 strategies")
    T, N = M.shape
    if n_splits % 2 != 0:
        n_splits -= 1
    n_splits = min(n_splits, T)
    if n_splits < 2:
        raise ValueError("need at least 2 time splits")

    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    logits = []

    for is_blocks in combinations(range(n_splits), half):
        is_set = set(is_blocks)
        is_idx = np.concatenate([blocks[b] for b in range(n_splits) if b in is_set])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])

        is_sr = np.array([periodic_sharpe(M[is_idx, j]) for j in range(N)])
        oos_sr = np.array([periodic_sharpe(M[oos_idx, j]) for j in range(N)])

        best = int(np.argmax(is_sr))
        # Out-of-sample relative rank of the IS-best strategy in [0, 1].
        rank = (np.sum(oos_sr <= oos_sr[best])) / N
        rank = min(max(rank, 1.0 / (N + 1)), 1.0 - 1.0 / (N + 1))
        logits.append(math.log(rank / (1.0 - rank)))

    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0.0))
    return {"pbo": pbo, "n_combinations": int(logits.size), "n_strategies": int(N)}
