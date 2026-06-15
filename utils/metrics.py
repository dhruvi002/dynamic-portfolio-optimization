"""
Portfolio Performance Metrics
===============================
Sharpe, Sortino, Calmar, Max Drawdown, and comparison helpers.
All functions accept daily return arrays (not log returns).
"""

import numpy as np
import pandas as pd
from typing import Union


TRADING_DAYS = 252


def compute_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free / TRADING_DAYS
    if excess.std() < 1e-10:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std())


def compute_sortino(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() < 1e-10:
        return float("inf")
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside.std())


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum drawdown as a fraction (negative number)."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / (peak + 1e-10)
    return float(drawdown.min())


def compute_calmar(returns: np.ndarray, portfolio_values: np.ndarray) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    ann_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    mdd = abs(compute_max_drawdown(portfolio_values))
    if mdd < 1e-10:
        return float("inf")
    return float(ann_return / mdd)


def compute_all_metrics(returns: np.ndarray, portfolio_values: np.ndarray) -> dict:
    """Compute the full suite of risk metrics."""
    return {
        "sharpe":         compute_sharpe(returns),
        "sortino":        compute_sortino(returns),
        "calmar":         compute_calmar(returns, portfolio_values),
        "max_drawdown":   compute_max_drawdown(portfolio_values),
        "total_return":   float((portfolio_values[-1] / portfolio_values[0]) - 1),
        "ann_return":     float((1 + returns.mean()) ** TRADING_DAYS - 1),
        "ann_volatility": float(returns.std() * np.sqrt(TRADING_DAYS)),
        "win_rate":       float((returns > 0).mean()),
        "final_value":    float(portfolio_values[-1]),
    }


def equal_weight_baseline(df: pd.DataFrame, tickers: list, initial_capital: float = 1e6):
    """
    Equal-weight buy-and-hold baseline (no costs, kept for backward compatibility).
    df must have columns: date, tic, close
    Returns (metrics_dict, port_values array, dates array)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="tic", values="close")
    available = [t for t in tickers if t in pivot.columns]
    pivot = pivot[available].dropna()
    daily_returns = pivot.pct_change().dropna()
    port_returns = daily_returns.mean(axis=1).values
    port_values = initial_capital * np.cumprod(1 + port_returns)
    port_values = np.insert(port_values, 0, initial_capital)
    dates = [pivot.index[0]] + list(daily_returns.index)
    metrics = compute_all_metrics(port_returns, port_values)
    return metrics, port_values, dates


_DISPLAY_KEYS = [
    "sharpe", "sortino", "calmar", "max_drawdown",
    "total_return", "ann_return", "ann_volatility", "win_rate",
]


def print_comparison(results: dict):
    """
    Print a multi-column comparison table.

    results: dict mapping strategy name → metrics dict.
             First key is the primary strategy (typically the SAC agent).
    """
    names = list(results.keys())
    col_w = 13

    header = f"  {'Metric':<18}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    for k in _DISPLAY_KEYS:
        row = f"  {k:<18}"
        for name in names:
            v = results[name].get(k, float("nan"))
            try:
                row += f"{v:>{col_w}.4f}"
            except (TypeError, ValueError):
                row += f"{'N/A':>{col_w}}"
        print(row)
    print(sep)
