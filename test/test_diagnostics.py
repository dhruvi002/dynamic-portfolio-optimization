"""
test/test_diagnostics.py — Phase 1 policy-behavior & aggregation diagnostics.
All inputs are analytically known.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils.diagnostics import (
    returns_from_values,
    reconcile_returns,
    turnover_series,
    hhi_series,
    active_share_series,
    bootstrap_ci,
    aggregate_metrics,
)


# ── return reconstruction ───────────────────────────────────────────────────────

def test_returns_from_values_known():
    v = np.array([100.0, 110.0, 99.0])
    r = returns_from_values(v)
    assert np.allclose(r, [0.10, -0.10], atol=1e-9)


def test_reconcile_flags_cost_drag():
    # Gross returns higher than the value-path (net) returns → positive cost drag.
    v = np.array([1_000_000.0, 1_005_000.0, 990_000.0, 995_000.0])
    gross = returns_from_values(v) + 0.0005      # pretend gross is higher
    out = reconcile_returns(gross, v)
    assert out["cost_drag_daily"] > 0
    assert abs(out["total_return"] - (v[-1] / v[0] - 1.0)) < 1e-12


# ── turnover ────────────────────────────────────────────────────────────────────

def test_turnover_known():
    # Two assets flipping fully: weights (0.5,0.5)->(1,0)->(0,1)
    w = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    t = turnover_series(w)
    assert np.allclose(t, [1.0, 2.0], atol=1e-9)


# ── HHI ─────────────────────────────────────────────────────────────────────────

def test_hhi_uniform_is_one_over_n():
    w = np.full((3, 4), 0.25)
    assert np.allclose(hhi_series(w), 0.25)        # 4 × 0.25² = 0.25 = 1/N


def test_hhi_concentrated_is_one():
    w = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert np.allclose(hhi_series(w), 1.0)


# ── active share ────────────────────────────────────────────────────────────────

def test_active_share_zero_for_equal_weight():
    w = np.full((2, 5), 0.2)
    assert np.allclose(active_share_series(w), 0.0, atol=1e-12)


def test_active_share_one_for_disjoint():
    # All weight on asset 0 vs equal-weight 1/4 → active share = 0.5*Σ|w-b|
    w = np.array([[1.0, 0.0, 0.0, 0.0]])
    a = active_share_series(w)
    expected = 0.5 * (abs(1 - 0.25) + 3 * abs(0 - 0.25))   # = 0.75
    assert np.allclose(a, expected)


# ── aggregation ─────────────────────────────────────────────────────────────────

def test_bootstrap_ci_single_value():
    lo, hi = bootstrap_ci(np.array([1.5]))
    assert lo == hi == 1.5


def test_aggregate_metrics_mean_and_ci():
    per_seed = [{"sharpe": 1.0}, {"sharpe": 2.0}, {"sharpe": 3.0}]
    agg = aggregate_metrics(per_seed, keys=["sharpe"], n_boot=2000, seed=0)
    assert abs(agg["sharpe"]["mean"] - 2.0) < 1e-9
    assert agg["sharpe"]["n"] == 3
    assert agg["sharpe"]["ci_low"] <= 2.0 <= agg["sharpe"]["ci_high"]


def test_aggregate_skips_nonfinite():
    per_seed = [{"sharpe": 1.0}, {"sharpe": float("inf")}, {"sharpe": 3.0}]
    agg = aggregate_metrics(per_seed, keys=["sharpe"])
    assert agg["sharpe"]["n"] == 2          # inf dropped
