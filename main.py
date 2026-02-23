#!/usr/bin/env python3
"""
main.py  —  Deep RL Portfolio Optimization
==========================================
Usage:
    python main.py --mode train
    python main.py --mode tune
    python main.py --mode backtest
    python main.py --mode train --episodes 200 --config tuning/best_config.json
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

from data.pipeline import download_data, add_technical_indicators, split_data
from environment.portfolio_env import PortfolioEnv
from agent.sac import SACAgent
from utils.trainer import train, backtest
from utils.metrics import equal_weight_baseline, print_comparison


# ─── Default hyperparameters ──────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "gamma":       0.99,
    "tau":         0.005,
    "lr_actor":    3e-4,
    "lr_critic":   3e-4,
    "lr_alpha":    3e-4,
    "batch_size":  256,
    "hidden_size": 256,
    "buffer_size": 1_000_000,
}


def build_env(df: pd.DataFrame) -> PortfolioEnv:
    return PortfolioEnv(
        df,
        transaction_cost_rate=0.001,
        slippage_rate=0.001,
        initial_capital=1_000_000.0,
    )


def build_agent(env: PortfolioEnv, config: dict) -> SACAgent:
    return SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=config["gamma"],
        tau=config["tau"],
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        lr_alpha=config["lr_alpha"],
        batch_size=int(config["batch_size"]),
        buffer_size=config.get("buffer_size", 1_000_000),
        hidden_sizes=[int(config.get("hidden_size", 256))] * 2,
    )


# ─── Modes ────────────────────────────────────────────────────────────────────

def mode_train(args, config: dict):
    print("=" * 60)
    print("  MODE: TRAIN")
    print("=" * 60)

    # Data
    data_path = "data/processed_data.parquet"
    if os.path.exists(data_path):
        print("Loading cached processed data…")
        df = pd.read_parquet(data_path)
    else:
        df = download_data(start="2015-01-01", end="2023-12-31")
        df = add_technical_indicators(df)
        os.makedirs("data", exist_ok=True)
        df.to_parquet(data_path, index=False)

    train_df, test_df = split_data(df)

    # Build env + agent
    train_env = build_env(train_df)
    agent = build_agent(train_env, config)
    print(f"  State dim: {train_env.state_dim}  |  Action dim: {train_env.action_dim}")
    print(f"  Device: {agent.device}")

    # Optional TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter("runs/sac_portfolio")
        print("  TensorBoard: runs/sac_portfolio")
    except ImportError:
        pass

    os.makedirs("checkpoints", exist_ok=True)
    logs = train(
        agent, train_env,
        n_episodes=args.episodes,
        save_path="checkpoints/best_agent.pt",
        writer=writer,
    )

    # Save training logs
    pd.DataFrame(logs).to_csv("checkpoints/training_log.csv", index=False)
    print("  Training log saved → checkpoints/training_log.csv")

    # Quick backtest on test set
    print("\nRunning backtest on test set…")
    test_env = build_env(test_df)
    agent.load("checkpoints/best_agent.pt")
    agent_metrics = backtest(agent, test_env)

    baseline_metrics, baseline_values, baseline_dates = equal_weight_baseline(
        test_df, PortfolioEnv.DJ30_TICKERS, initial_capital=1_000_000.0
    )

    print("\nResults vs. Equal-Weight Baseline:")
    print_comparison(agent_metrics, baseline_metrics)

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_training_curves, plot_backtest_comparison, plot_weight_heatmap
        os.makedirs("plots", exist_ok=True)

        import pandas as _pd_
        log_df = _pd_.read_csv("checkpoints/training_log.csv")
        plot_training_curves(log_df, save_path="plots/training_curves.png")

        agent_values = np.array(test_env.history["portfolio_value"])
        plot_backtest_comparison(agent_values, baseline_values, baseline_dates,
                                 save_path="plots/backtest_comparison.png")

        weights_history = np.array(test_env.history["weights"])
        plot_weight_heatmap(weights_history, PortfolioEnv.DJ30_TICKERS,
                            save_path="plots/weight_heatmap.png")
        print("  Plots saved to plots/")
    except Exception as e:
        print(f"  Plot generation skipped: {e}")


def mode_tune(args):
    print("=" * 60)
    print("  MODE: RAY TUNE HPO (50 trials)")
    print("=" * 60)
    from tuning.tune_runner import run_tune
    best = run_tune(num_samples=args.tune_samples)
    os.makedirs("tuning", exist_ok=True)
    with open("tuning/best_config.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best config saved → tuning/best_config.json")
    return best


def mode_backtest(args, config: dict):
    print("=" * 60)
    print("  MODE: BACKTEST")
    print("=" * 60)

    df = pd.read_parquet("data/processed_data.parquet")
    _, test_df = split_data(df, train_start="2019-04-01", train_end="2022-12-31",
                            test_start="2023-01-01", test_end="2025-01-31")
    test_env = build_env(test_df)
    agent = build_agent(test_env, config)
    agent.load(args.checkpoint)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    agent_metrics = backtest(agent, test_env)
    baseline_metrics, _, _ = equal_weight_baseline(
        test_df, PortfolioEnv.DJ30_TICKERS, initial_capital=1_000_000.0
    )
    print("\nBacktest Results vs. Equal-Weight Baseline:")
    print_comparison(agent_metrics, baseline_metrics)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep RL Portfolio Optimization")
    parser.add_argument("--mode", choices=["train", "tune", "backtest"], default="train")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--tune-samples", type=int, default=50)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config (from Ray Tune or manual)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_agent.pt")
    args = parser.parse_args()

    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(json.load(f))
        print(f"Loaded config from {args.config}")

    if args.mode == "train":
        mode_train(args, config)
    elif args.mode == "tune":
        best = mode_tune(args)
        # Optionally chain into training with best config
        config.update(best)
        mode_train(args, config)
    elif args.mode == "backtest":
        mode_backtest(args, config)


if __name__ == "__main__":
    main()
