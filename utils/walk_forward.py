"""
Walk-forward cross-validation for time-series portfolio models.
=================================================================
Uses an expanding training window: each fold trains on all data from the
start up to the fold boundary, then evaluates on the next ``test_months`` of
unseen data. This gives an unbiased variance estimate across market regimes
without ever leaking future data into the training window.

Phase 3 changes
---------------
* **NET-of-cost metrics (Task A).** ``walk_forward`` previously reported the
  *gross* Sharpe/Sortino/Calmar straight from ``trainer.backtest()`` — the exact
  bug Phase 1 fixed for the single-window harness. Each fold now recomputes
  Sharpe/Sortino/Calmar/total_return from the **net-of-cost** value path
  (``diagnostics.returns_from_values``) and exposes ``gross_sharpe`` alongside so
  the transaction-cost drag is visible. This mirrors
  ``experiments/multi_seed._net_metrics``.
* **Per-fold policy diagnostics.** Turnover / HHI / active-share via
  ``diagnostics.policy_diagnostics`` so the regime table can attribute net
  underperformance to turnover (the binding constraint identified in Phase 2).
* **Leak assertion.** Every fold asserts ``train_end < test_start`` (walk-forward
  is leak-free by construction, but the guard documents and enforces it).
* **Net return series per fold.** Each fold dict carries ``agent_returns`` (net
  daily) and ``test_dates`` so the Phase-3 harness can run a per-fold
  Jobson–Korkie–Memmel test vs an equal-weight baseline over the same window.
* **Testable scheduling.** Fold windows are produced by ``_fold_windows`` so the
  chronology / non-overlap / expanding-train / clean-stop properties can be unit
  tested without any training.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional
from pandas.tseries.offsets import DateOffset

from utils.diagnostics import returns_from_values, policy_diagnostics


# ── Fold scheduling (pure, unit-testable) ───────────────────────────────────────

def _fold_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_folds: int,
    test_months: int,
    min_train_months: int,
) -> list:
    """
    Compute the (train_end, test_start, test_end) windows for an expanding
    walk-forward, without touching any data or model.

    The training window is implicitly ``[start_date, train_end]`` and *expands*
    fold over fold; consecutive test windows are contiguous and non-overlapping.
    A fold whose test window would run past ``end_date`` is dropped (clean stop),
    so the returned list may be shorter than ``n_folds``.

    Returns a list of dicts with Timestamp fields ``train_end``, ``test_start``,
    ``test_end`` and the integer ``fold`` (1-based).
    """
    first_test_start = start_date + DateOffset(months=min_train_months)
    windows = []
    for fold_idx in range(n_folds):
        test_start = first_test_start + DateOffset(months=fold_idx * test_months)
        test_end = test_start + DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > end_date:
            break
        train_end = test_start - pd.Timedelta(days=1)
        # Leak-free by construction; enforce + document the invariant.
        assert train_end < test_start, (
            f"WALK-FORWARD LEAK GUARD: fold {fold_idx + 1} has "
            f"train_end {train_end.date()} >= test_start {test_start.date()}"
        )
        windows.append({
            "fold": fold_idx + 1,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
    return windows


# ── Net-of-cost fold metrics ─────────────────────────────────────────────────────

def _fold_net_metrics(test_env) -> dict:
    """
    Compute a fold's headline metrics from the **net-of-cost** value path
    (consistent with the equity curve, total_return, and the JK test), and
    expose the **gross** Sharpe alongside so the transaction-cost impact is
    visible. ``backtest()``'s own metrics use the env's gross return array and
    so overstate Sharpe/Sortino/Calmar whenever turnover (hence costs) is high.

    Must be called *after* a deterministic ``backtest`` has populated
    ``test_env.history``.
    """
    from utils.metrics import compute_all_metrics, compute_sharpe
    pv = np.asarray(test_env.history["portfolio_value"], dtype=float)
    net = returns_from_values(pv)                                  # net of cost
    gross = np.asarray(test_env.history["returns"], dtype=float)   # gross of cost
    m = compute_all_metrics(net, pv)             # net Sharpe/Sortino/Calmar/…
    m["gross_sharpe"] = compute_sharpe(gross)    # annualized, pre-cost
    m["final_value"] = float(pv[-1])
    return m


def _agent_net_returns(test_env) -> tuple:
    """Net daily return series + aligned dates (as 'YYYY-MM-DD' strings)."""
    pv = np.asarray(test_env.history["portfolio_value"], dtype=float)
    rets = returns_from_values(pv)
    dates = pd.to_datetime(list(test_env.dates))[1:1 + len(rets)]
    return rets, [str(d.date()) for d in dates]


# ── Walk-forward driver ──────────────────────────────────────────────────────────

def walk_forward(
    df: pd.DataFrame,
    agent_factory: Callable,
    normalizer_factory: Callable,
    n_folds: int = 5,
    test_months: int = 6,
    min_train_months: int = 18,
    train_episodes: int = 50,
    warmup_steps: int = 500,
    env_kwargs: Optional[dict] = None,
) -> list:
    """
    Expanding-window walk-forward cross-validation.

    Slides forward across ``df`` (pass the full chronological span, e.g.
    TRAIN_START → TEST_END). Each fold trains on all preceding data and
    evaluates on the next ``test_months`` of unseen data.

    agent_factory(state_dim, action_dim) → SACAgent
    normalizer_factory(state_dim, n_skip) → RunningNormalizer

    Returns a list of per-fold metric dicts. Each carries the date range, the
    **net-of-cost** Sharpe/Sortino/Calmar/total_return (key ``sharpe`` is NET),
    ``gross_sharpe``, policy diagnostics (turnover/HHI/active-share), and the
    raw net daily return series (``agent_returns``) with aligned ``test_dates``.
    """
    from environment.portfolio_env import PortfolioEnv
    from utils.trainer import train, backtest

    kw = env_kwargs or {
        "transaction_cost_rate": 0.001,
        "slippage_rate": 0.001,
        "initial_capital": 1_000_000.0,
    }

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    start_date = df["date"].min()
    end_date = df["date"].max()

    windows = _fold_windows(start_date, end_date, n_folds, test_months, min_train_months)
    if len(windows) < n_folds:
        print(f"  Walk-forward: data window {start_date.date()} → {end_date.date()} "
              f"yields {len(windows)} usable folds (requested {n_folds}).")

    fold_results = []
    for w in windows:
        fold_idx = w["fold"]
        train_end, test_start, test_end = w["train_end"], w["test_start"], w["test_end"]

        # Leak guard (redundant with _fold_windows, kept explicit per Phase-3 spec).
        assert train_end < test_start, (
            f"WALK-FORWARD LEAK GUARD: fold {fold_idx} train_end {train_end.date()} "
            f">= test_start {test_start.date()}"
        )

        train_df = df[df["date"] <= train_end].copy()
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        train_env = PortfolioEnv(train_df, **kw)
        test_env = PortfolioEnv(test_df, **kw)

        agent = agent_factory(train_env.state_dim, train_env.action_dim)
        normalizer = normalizer_factory(train_env.state_dim, train_env.n_assets)

        print(f"\n  Fold {fold_idx}/{n_folds}: "
              f"train {start_date.date()} → {train_end.date()}, "
              f"test {test_start.date()} → {test_end.date()}")

        train(
            agent, train_env,
            n_episodes=train_episodes,
            warmup_steps=warmup_steps,
            normalizer=normalizer,
            save_path=None,
            log_every=train_episodes,   # only print at end of each fold
        )

        backtest(agent, test_env, normalizer=normalizer)   # populates test_env.history
        metrics = _fold_net_metrics(test_env)              # NET (+ gross_sharpe)
        pdiag = policy_diagnostics(test_env)
        agent_rets, test_dates = _agent_net_returns(test_env)

        metrics.update({
            "fold":          fold_idx,
            "train_start":   str(start_date.date()),
            "train_end":     str(train_end.date()),
            "test_start":    str(test_start.date()),
            "test_end":      str(test_end.date()),
            "n_test_days":   int(agent_rets.size),
            "mean_turnover": pdiag["mean_turnover"],
            "mean_hhi":      pdiag["mean_hhi"],
            "mean_active_share": pdiag["mean_active_share"],
            "near_uniform":  pdiag["near_uniform"],
            # raw net series for the harness's per-fold JK test (not for CSV)
            "agent_returns": agent_rets,
            "test_dates":    test_dates,
        })
        fold_results.append(metrics)
        print(f"  Fold {fold_idx} → NET Sharpe: {metrics['sharpe']:+.3f}  "
              f"(gross {metrics['gross_sharpe']:+.3f})  "
              f"MDD: {metrics['max_drawdown']:.2%}  "
              f"Return: {metrics['total_return']:+.2%}  "
              f"turnover: {metrics['mean_turnover']:.3f}")

    return fold_results


def walk_forward_summary(fold_results: list, keys: Optional[list] = None) -> dict:
    """
    Print and return mean ± std of metrics across folds.

    Returns dict of {metric: {'mean': float, 'std': float, 'n': int}}.
    """
    if not fold_results:
        print("  Walk-forward: no fold results to summarise.")
        return {}

    keys = keys or ["sharpe", "gross_sharpe", "sortino", "calmar",
                    "max_drawdown", "total_return", "mean_turnover"]
    summary = {}

    print("\nWalk-Forward Cross-Validation Summary (NET of cost)")
    print("─" * 52)
    print(f"  {'Metric':<18}  {'Mean':>8}  {'Std':>8}  {'Folds':>6}")
    print("─" * 52)

    for k in keys:
        vals = [r[k] for r in fold_results if k in r and np.isfinite(r[k])]
        if not vals:
            continue
        mean, std = float(np.mean(vals)), float(np.std(vals))
        summary[k] = {"mean": mean, "std": std, "n": len(vals)}
        print(f"  {k:<18}  {mean:>+8.3f}  {std:>8.3f}  {len(vals):>6}")

    print("─" * 52)
    return summary
