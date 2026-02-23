"""
Plotting utilities for portfolio analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def plot_training_curves(log_df: pd.DataFrame, save_path: str = None):
    """Plot Sharpe, return, drawdown, and alpha over training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("SAC Training Curves", fontsize=16, fontweight="bold")

    metrics = [
        ("sharpe", "Sharpe Ratio", "steelblue"),
        ("total_return", "Episode Return", "seagreen"),
        ("max_drawdown", "Max Drawdown", "crimson"),
        ("alpha", "Entropy α", "darkorchid"),
    ]

    for ax, (col, label, color) in zip(axes.flat, metrics):
        if col in log_df.columns:
            ax.plot(log_df["episode"], log_df[col], color=color, lw=1.5, alpha=0.8)
            # Rolling mean
            rolling = log_df[col].rolling(10, min_periods=1).mean()
            ax.plot(log_df["episode"], rolling, color=color, lw=2.5, label="10-ep MA")
            ax.set_title(label)
            ax.set_xlabel("Episode")
            ax.legend()
            ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_backtest_comparison(
    agent_values: np.ndarray,
    baseline_values: np.ndarray,
    dates,
    agent_label: str = "SAC Agent",
    baseline_label: str = "Equal-Weight",
    save_path: str = None,
):
    """Portfolio value curves + drawdown comparison."""
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

    # ── Value curves ─────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(dates[:len(agent_values)], agent_values / agent_values[0],
             label=agent_label, color="steelblue", lw=2)
    ax0.plot(dates[:len(baseline_values)], baseline_values / baseline_values[0],
             label=baseline_label, color="coral", lw=2, linestyle="--")
    ax0.set_title("Portfolio Value (Normalised)", fontsize=13, fontweight="bold")
    ax0.legend()
    ax0.grid(alpha=0.3)
    ax0.set_ylabel("Normalised Value")

    # ── Drawdown ─────────────────────────────────────────────────────────────
    def drawdown_series(values):
        peak = np.maximum.accumulate(values)
        return (values - peak) / (peak + 1e-10)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    agent_dd = drawdown_series(agent_values)
    baseline_dd = drawdown_series(baseline_values)
    ax1.fill_between(dates[:len(agent_dd)], agent_dd, alpha=0.4, color="steelblue",
                     label="SAC DD")
    ax1.fill_between(dates[:len(baseline_dd)], baseline_dd, alpha=0.4, color="coral",
                     label="Baseline DD")
    ax1.set_ylabel("Drawdown")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Rolling Sharpe ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    window = 63  # ~3 months
    agent_ret = np.diff(agent_values) / (agent_values[:-1] + 1e-10)
    roll_sharpe = pd.Series(agent_ret).rolling(window).apply(
        lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(252)
    )
    ax2.plot(dates[1:len(roll_sharpe)+1], roll_sharpe, color="steelblue", lw=1.5)
    ax2.axhline(0, color="gray", linestyle=":", lw=1)
    ax2.set_ylabel(f"Rolling Sharpe ({window}d)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_weight_heatmap(weights_history: np.ndarray, tickers: list, save_path: str = None):
    """Heatmap of portfolio weights over time."""
    fig, ax = plt.subplots(figsize=(16, 6))
    weights_df = pd.DataFrame(weights_history, columns=tickers)
    sns.heatmap(weights_df.T, ax=ax, cmap="YlOrRd", vmin=0, vmax=0.2,
                xticklabels=max(1, len(weights_history) // 20),
                cbar_kws={"label": "Weight"})
    ax.set_title("Portfolio Weight Allocation Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Asset")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
