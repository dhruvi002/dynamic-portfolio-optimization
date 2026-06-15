"""
test/test_metrics.py - Performance metric unit tests
All inputs are analytically known — no randomness.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import pytest

from utils.metrics import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_all_metrics,
)


# ── Sharpe ────────────────────────────────────────────────────────────────────

def test_sharpe_known():
    # Constant returns → std=0 → zero-variance guard returns 0.0.
    returns_flat = np.ones(252) * 0.001
    assert compute_sharpe(returns_flat) == 0.0

    # Positive mean with non-zero variance → Sharpe > 0.
    returns_vary = np.full(252, 0.001) + np.linspace(-0.0005, 0.0005, 252)
    assert compute_sharpe(returns_vary) > 0


def test_sharpe_negative():
    # Negative mean returns → negative Sharpe.
    returns = np.full(252, -0.001) + np.linspace(-0.0005, 0.0005, 252)
    assert compute_sharpe(returns) < 0


# ── Sortino ───────────────────────────────────────────────────────────────────

def test_sortino_ignores_upside():
    # All returns non-negative → no downside deviation → Sortino = inf.
    returns = np.full(100, 0.01)
    assert math.isinf(compute_sortino(returns))


# ── Max drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_known():
    # Peak at 110, trough at 90 → MDD = (90 - 110) / 110 ≈ -0.1818.
    values = np.array([100.0, 110.0, 105.0, 90.0, 95.0, 100.0])
    mdd = compute_max_drawdown(values)
    expected = (90.0 - 110.0) / 110.0  # ≈ -0.18182
    assert abs(mdd - expected) < 1e-4, f"mdd={mdd:.6f}, expected≈{expected:.6f}"


def test_max_drawdown_monotone():
    # Always-rising portfolio → no drawdown.
    values = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
    assert compute_max_drawdown(values) == 0.0


# ── Calmar ────────────────────────────────────────────────────────────────────

def test_calmar_known():
    # Portfolio: 1.0 → 1.2 → 1.0
    # MDD = (1.0 - 1.2) / 1.2 = -1/6; returns match value transitions.
    values  = np.array([1.0, 1.2, 1.0])
    returns = np.array([0.2, -1.0 / 6.0])

    mdd_abs    = abs(compute_max_drawdown(values))
    ann_return = (1.0 + np.mean(returns)) ** 252 - 1.0
    expected   = ann_return / mdd_abs

    result = compute_calmar(returns, values)
    assert abs(result - expected) < 1e-6, f"calmar={result}, expected={expected}"


# ── Total return ──────────────────────────────────────────────────────────────

def test_total_return_known():
    # Portfolio grows from 1 000 000 to 1 100 000 → total return = 10 %.
    values  = np.array([1_000_000.0, 1_100_000.0])
    returns = np.array([0.1])
    metrics = compute_all_metrics(returns, values)
    assert abs(metrics["total_return"] - 0.10) < 1e-6
