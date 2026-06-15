"""
Walk-forward cross-validation for time-series portfolio models.

Uses an expanding training window: each fold trains on all data from the
start up to the fold boundary, then evaluates on the next test_months of
unseen data. This gives an unbiased variance estimate across market regimes
without ever leaking future data into the training window.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional
from pandas.tseries.offsets import DateOffset


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

    Slides forward across df (pass the combined train+val window, e.g.
    TRAIN_START → TRAIN_END). Each fold trains on all preceding data and
    evaluates on the next test_months of unseen data.

    agent_factory(state_dim, action_dim) → SACAgent
    normalizer_factory(state_dim, n_skip) → RunningNormalizer

    Returns list of per-fold metric dicts (keys include 'fold', date range
    strings, and all metrics from compute_all_metrics).
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
    end_date   = df["date"].max()

    first_test_start = start_date + DateOffset(months=min_train_months)
    fold_results = []

    for fold_idx in range(n_folds):
        test_start = first_test_start + DateOffset(months=fold_idx * test_months)
        test_end   = test_start + DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_end > end_date:
            print(f"  Walk-forward: fold {fold_idx + 1} would exceed data window "
                  f"({test_end.date()} > {end_date.date()}), stopping at {fold_idx} folds.")
            break

        train_end = test_start - pd.Timedelta(days=1)
        train_df  = df[df["date"] <= train_end].copy()
        test_df   = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        train_env = PortfolioEnv(train_df, **kw)
        test_env  = PortfolioEnv(test_df, **kw)

        agent      = agent_factory(train_env.state_dim, train_env.action_dim)
        normalizer = normalizer_factory(train_env.state_dim, train_env.n_assets)

        print(f"\n  Fold {fold_idx + 1}/{n_folds}: "
              f"train {start_date.date()} → {train_end.date()}, "
              f"test {test_start.date()} → {test_end.date()}")

        train(
            agent, train_env,
            n_episodes=train_episodes,
            warmup_steps=warmup_steps,
            normalizer=normalizer,
            save_path=None,
            log_every=train_episodes,  # only print at end of each fold
        )

        metrics = backtest(agent, test_env, normalizer=normalizer)
        metrics.update({
            "fold":        fold_idx + 1,
            "train_start": str(start_date.date()),
            "train_end":   str(train_end.date()),
            "test_start":  str(test_start.date()),
            "test_end":    str(test_end.date()),
        })
        fold_results.append(metrics)
        print(f"  Fold {fold_idx + 1} → Sharpe: {metrics['sharpe']:.3f}  "
              f"MDD: {metrics['max_drawdown']:.2%}  "
              f"Return: {metrics['total_return']:+.2%}")

    return fold_results


def walk_forward_summary(fold_results: list, keys: Optional[list] = None) -> dict:
    """
    Print and return mean ± std of metrics across folds.

    Returns dict of {metric: {'mean': float, 'std': float}}.
    """
    if not fold_results:
        print("  Walk-forward: no fold results to summarise.")
        return {}

    keys = keys or ["sharpe", "sortino", "calmar", "max_drawdown", "total_return"]
    summary = {}

    print("\nWalk-Forward Cross-Validation Summary")
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
