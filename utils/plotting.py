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


def plot_alpha_entropy(log_df: pd.DataFrame, save_path: str = None):
    """
    Plot the entropy temperature alpha (log scale) and the mean policy entropy
    over training. Documents the entropy collapse diagnosed in Phase 1 (I-6).
    """
    fig, ax1 = plt.subplots(figsize=(11, 5))
    fig.suptitle("Entropy Temperature & Policy Entropy over Training",
                 fontsize=14, fontweight="bold")

    if "alpha" in log_df.columns:
        ax1.plot(log_df["episode"], log_df["alpha"].clip(lower=1e-30),
                 color="darkorchid", lw=1.8, label="α (entropy temp)")
        ax1.set_yscale("log")
        ax1.set_ylabel("α  (log scale)", color="darkorchid")
        ax1.tick_params(axis="y", labelcolor="darkorchid")
    ax1.set_xlabel("Episode")
    ax1.grid(alpha=0.3)

    if "policy_entropy" in log_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(log_df["episode"], log_df["policy_entropy"],
                 color="seagreen", lw=1.8, label="policy entropy −E[log π]")
        ax2.set_ylabel("Policy entropy", color="seagreen")
        ax2.tick_params(axis="y", labelcolor="seagreen")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_diagnostics_panel(
    oos_diag: dict,
    is_diag: dict = None,
    save_path: str = None,
):
    """
    One-page policy-behavior diagnostic figure (Phase 1, resolves I-5).

    Four panels:
      (1) Turnover Σ|Δw| per step (OOS, with IS overlay if provided).
      (2) Weight concentration HHI per step vs the 1/N uniform floor.
      (3) Active share vs equal-weight per step.
      (4) In-sample vs out-of-sample equity curves (normalised) — the figure
          that reconciles the IS/OOS gap.

    `oos_diag` / `is_diag` are dicts returned by
    `utils.diagnostics.policy_diagnostics`.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Phase 1 — Policy-Behavior Diagnostics", fontsize=15, fontweight="bold")

    def _x(series):
        return np.arange(len(series))

    # (1) Turnover
    ax = axes[0, 0]
    ax.plot(_x(oos_diag["turnover_series"]), oos_diag["turnover_series"],
            color="steelblue", lw=1.2, label=f"OOS (mean {oos_diag['mean_turnover']:.3f})")
    if is_diag is not None:
        ax.plot(_x(is_diag["turnover_series"]), is_diag["turnover_series"],
                color="coral", lw=1.0, alpha=0.7,
                label=f"IS (mean {is_diag['mean_turnover']:.3f})")
    ax.set_title("Turnover  Σ|Δw|  per step")
    ax.set_xlabel("Step"); ax.set_ylabel("Turnover"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2) HHI concentration
    ax = axes[0, 1]
    ax.plot(_x(oos_diag["hhi_series"]), oos_diag["hhi_series"],
            color="steelblue", lw=1.2, label=f"OOS HHI (mean {oos_diag['mean_hhi']:.4f})")
    if is_diag is not None:
        ax.plot(_x(is_diag["hhi_series"]), is_diag["hhi_series"],
                color="coral", lw=1.0, alpha=0.7, label="IS HHI")
    ax.axhline(oos_diag["hhi_uniform_floor"], color="gray", ls=":", lw=1.2,
               label=f"1/N uniform = {oos_diag['hhi_uniform_floor']:.4f}")
    ax.set_title("Weight Concentration (HHI = Σ w²)")
    ax.set_xlabel("Step"); ax.set_ylabel("HHI"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3) Active share
    ax = axes[1, 0]
    ax.plot(_x(oos_diag["active_share_series"]), oos_diag["active_share_series"],
            color="steelblue", lw=1.2, label=f"OOS (mean {oos_diag['mean_active_share']:.3f})")
    if is_diag is not None:
        ax.plot(_x(is_diag["active_share_series"]), is_diag["active_share_series"],
                color="coral", lw=1.0, alpha=0.7, label="IS")
    ax.set_title("Active Share vs Equal-Weight  (0.5·Σ|w−1/N|)")
    ax.set_xlabel("Step"); ax.set_ylabel("Active share"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (4) IS vs OOS equity curves (normalised)
    ax = axes[1, 1]
    oos_v = oos_diag["portfolio_values"]
    ax.plot(np.linspace(0, 1, len(oos_v)), oos_v / oos_v[0],
            color="steelblue", lw=2, label="Out-of-sample (test)")
    if is_diag is not None:
        is_v = is_diag["portfolio_values"]
        ax.plot(np.linspace(0, 1, len(is_v)), is_v / is_v[0],
                color="coral", lw=2, ls="--", label="In-sample (train)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_title("Equity Curve: In-Sample vs Out-of-Sample (normalised)")
    ax.set_xlabel("Normalised time"); ax.set_ylabel("Value / start"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_walk_forward_regimes(regime_agg: dict, save_path: str = None):
    """
    Per-regime NET Sharpe of the SAC agent vs the equal-weight baseline, with
    bootstrap 95% CI error bars (Phase 3). `regime_agg` is
    {regime: aggregate_metrics(...)} as produced by walk_forward_eval; each entry
    must hold 'sharpe' and 'ew_sharpe' stats dicts with mean/ci_low/ci_high.
    """
    regimes = list(regime_agg.keys())
    if not regimes:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no regimes to plot", ha="center", va="center")
        return fig

    x = np.arange(len(regimes))
    w = 0.38

    def _series(key):
        means, los, his = [], [], []
        for rg in regimes:
            s = regime_agg[rg].get(key, {})
            m = s.get("mean", np.nan)
            means.append(m)
            los.append(m - s.get("ci_low", m))
            his.append(s.get("ci_high", m) - m)
        return np.array(means), np.abs(np.array([los, his]))

    agent_m, agent_err = _series("sharpe")
    ew_m, ew_err = _series("ew_sharpe")

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(regimes) + 4), 5))
    ax.bar(x - w / 2, agent_m, w, yerr=agent_err, capsize=4,
           color="steelblue", label="SAC agent (net)", alpha=0.9)
    ax.bar(x + w / 2, ew_m, w, yerr=ew_err, capsize=4,
           color="darkorange", label="Equal-weight (net)", alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_ylabel("Annualized NET Sharpe")
    ax.set_title("Walk-Forward: NET Sharpe by Market Regime (95% CI)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
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
