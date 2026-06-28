"""
test/test_significance.py — Phase 1 significance & overfitting diagnostics.
Uses analytically-known or constructed cases so results are deterministic.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils.significance import (
    periodic_sharpe,
    annualize_sharpe,
    jobson_korkie_memmel,
    sharpe_diff_bootstrap_ci,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


# ── basic Sharpe helpers ────────────────────────────────────────────────────────

def test_periodic_sharpe_zero_variance():
    assert periodic_sharpe(np.ones(50)) == 0.0


def test_annualize():
    assert abs(annualize_sharpe(1.0, 252) - np.sqrt(252)) < 1e-9


# ── Jobson–Korkie–Memmel ────────────────────────────────────────────────────────

def test_jk_identical_series_zero_diff():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    out = jobson_korkie_memmel(r, r)
    assert abs(out["sharpe_diff_periodic"]) < 1e-12
    assert out["p_value"] > 0.99          # no difference → cannot reject H0
    assert not out["significant_5pct"]
    assert abs(out["correlation"] - 1.0) < 1e-9


def test_jk_clear_difference_is_significant():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.01, 2000)
    a = base + 0.0015          # same noise, higher mean → higher Sharpe
    b = base
    out = jobson_korkie_memmel(a, b)
    assert out["sharpe_diff_periodic"] > 0
    assert out["p_value"] < 0.05
    assert out["significant_5pct"]


def test_jk_requires_alignment():
    with pytest.raises(ValueError):
        jobson_korkie_memmel(np.zeros(10), np.zeros(11))


# ── bootstrap CI ────────────────────────────────────────────────────────────────

def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 0.01, 1500)
    a = base + 0.0012
    b = base
    out = sharpe_diff_bootstrap_ci(a, b, n_boot=2000, seed=0)
    lo, hi = out["ci_periodic"]
    assert lo <= out["sharpe_diff_periodic"] <= hi
    assert out["ci_excludes_zero"]        # a clearly beats b
    assert out["sharpe_diff_annual"] > 0


def test_bootstrap_ci_includes_zero_for_identical():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, 1000)
    out = sharpe_diff_bootstrap_ci(r, r.copy(), n_boot=1000, seed=0)
    lo, hi = out["ci_periodic"]
    assert lo <= 0 <= hi
    assert not out["ci_excludes_zero"]


# ── PSR / DSR ───────────────────────────────────────────────────────────────────

def test_psr_increases_with_sample_length():
    # Same Sharpe, more observations → more confidence it beats benchmark.
    p_short = probabilistic_sharpe_ratio(0.1, 50, 0.0, 3.0)
    p_long = probabilistic_sharpe_ratio(0.1, 5000, 0.0, 3.0)
    assert 0.0 <= p_short <= 1.0 and 0.0 <= p_long <= 1.0
    assert p_long > p_short


def test_expected_max_sharpe_grows_with_trials():
    s10 = expected_max_sharpe(10, 0.1)
    s1000 = expected_max_sharpe(1000, 0.1)
    assert s1000 > s10 > 0


def test_dsr_haircut_with_more_trials():
    # More trials → higher deflation benchmark → lower DSR for the same Sharpe.
    d_few = deflated_sharpe_ratio(0.12, 500, 0.0, 3.0, n_trials=5, sr_trials_std=0.1)
    d_many = deflated_sharpe_ratio(0.12, 500, 0.0, 3.0, n_trials=500, sr_trials_std=0.1)
    assert d_many["dsr"] < d_few["dsr"]
    assert 0.0 <= d_many["dsr"] <= 1.0


# ── PBO ─────────────────────────────────────────────────────────────────────────

def test_pbo_in_unit_interval_for_noise():
    rng = np.random.default_rng(4)
    M = rng.normal(0.0, 0.01, size=(240, 8))
    out = probability_of_backtest_overfitting(M, n_splits=8)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["n_strategies"] == 8


def test_pbo_lower_when_one_strategy_dominates():
    # A genuinely superior strategy generalises OOS → lower PBO than pure noise.
    rng = np.random.default_rng(5)
    noise = rng.normal(0.0, 0.01, size=(240, 6))
    pbo_noise = probability_of_backtest_overfitting(noise.copy(), n_splits=8)["pbo"]

    dominant = noise.copy()
    dominant[:, 0] += 0.004                # column 0 is consistently better
    pbo_dom = probability_of_backtest_overfitting(dominant, n_splits=8)["pbo"]

    assert pbo_dom < 0.25
    assert pbo_dom <= pbo_noise            # dominance reduces overfitting probability
