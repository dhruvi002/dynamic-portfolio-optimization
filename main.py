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
import pickle
import numpy as np
import pandas as pd

from config import (DOWNLOAD_START, TRAIN_START, TRAIN_PROPER_END,
                    VAL_START, TRAIN_END, TEST_START, TEST_END)
from data.pipeline import download_data, add_technical_indicators, split_data, three_way_split
from environment.portfolio_env import PortfolioEnv
from agent.sac import SACAgent
from utils.trainer import train, backtest
from utils.metrics import print_comparison
from utils.normalizer import RunningNormalizer


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

NORMALIZER_PATH = "checkpoints/normalizer.pkl"


def build_env(df: pd.DataFrame, sentiment_df: pd.DataFrame = None) -> PortfolioEnv:
    return PortfolioEnv(
        df,
        transaction_cost_rate=0.001,
        slippage_rate=0.001,
        initial_capital=1_000_000.0,
        sentiment_df=sentiment_df,
    )


def build_agent(env: PortfolioEnv, config: dict, encoder: str = "mlp") -> SACAgent:
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
        encoder=encoder,
    )


def _load_sentiment(sentiment_path: str) -> pd.DataFrame:
    """Load precomputed sentiment scores; return None if file doesn't exist."""
    try:
        from data.sentiment_pipeline import load_precomputed
        return load_precomputed(sentiment_path)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        return None


def _load_data(data_path: str) -> pd.DataFrame:
    if os.path.exists(data_path):
        print("Loading cached processed data…")
        return pd.read_parquet(data_path)
    df = download_data(start=DOWNLOAD_START, end=TEST_END)
    df = add_technical_indicators(df)
    os.makedirs("data", exist_ok=True)
    df.to_parquet(data_path, index=False)
    return df


# ─── Modes ────────────────────────────────────────────────────────────────────

def _collect_baselines(test_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    """Run all baseline strategies and return {name: metrics_dict}."""
    from utils.baselines import equal_weight, spy_qqq, momentum_12_1, min_variance, max_sharpe_mvo
    tickers = PortfolioEnv.DJ30_TICKERS
    results = {}

    try:
        results["Equal Weight"], _, _ = equal_weight(test_df, tickers)
    except Exception as e:
        print(f"  Baseline 'Equal Weight' failed: {e}")

    try:
        results["SPY/QQQ 60/40"], _, _ = spy_qqq(start=TEST_START, end=TEST_END)
    except Exception as e:
        print(f"  Baseline 'SPY/QQQ' failed: {e}")

    try:
        results["Momentum 12-1"], _, _ = momentum_12_1(test_df, tickers)
    except Exception as e:
        print(f"  Baseline 'Momentum 12-1' failed: {e}")

    try:
        results["Min Variance"], _, _ = min_variance(test_df, tickers, train_df=train_df)
    except Exception as e:
        print(f"  Baseline 'Min Variance' failed: {e}")

    try:
        results["Max Sharpe MVO"], _, _ = max_sharpe_mvo(test_df, tickers, train_df=train_df)
    except Exception as e:
        print(f"  Baseline 'Max Sharpe MVO' failed: {e}")

    return results


def mode_train(args, config: dict):
    print("=" * 60)
    print("  MODE: TRAIN")
    print("=" * 60)

    df = _load_data("data/processed_data.parquet")

    sentiment_df = None
    if args.sentiment:
        sentiment_df = _load_sentiment(args.sentiment_path)
        if sentiment_df is not None:
            from data.sentiment_pipeline import merge_sentiment
            df = merge_sentiment(df, sentiment_df)
            print(f"  Sentiment loaded from {args.sentiment_path}")
        else:
            print("  Sentiment disabled (file not found).")

    # Chronological three-way split: train / val (held-out for checkpointing) / test
    train_df, val_df, test_df = three_way_split(
        df,
        train_start=TRAIN_START,  train_end=TRAIN_PROPER_END,
        val_start=VAL_START,      val_end=TRAIN_END,
        test_start=TEST_START,    test_end=TEST_END,
    )

    # Pass the pre-merged df to env; env reads 'sentiment_score' column if present
    merged_sentiment_df = None
    if args.sentiment and sentiment_df is not None:
        merged_sentiment_df = sentiment_df

    train_env = build_env(train_df, sentiment_df=merged_sentiment_df)
    val_env   = build_env(val_df, sentiment_df=merged_sentiment_df)
    agent     = build_agent(train_env, config, encoder=args.encoder)
    print(f"  State dim: {train_env.state_dim}  |  Action dim: {train_env.action_dim}")
    print(f"  Device: {agent.device}")

    # Normalizer: skip first n_assets dims (portfolio weights are already on simplex)
    normalizer = RunningNormalizer(train_env.state_dim, n_skip=train_env.n_assets)

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
        normalizer=normalizer,
        val_env=val_env,
    )

    pd.DataFrame(logs).to_csv("checkpoints/training_log.csv", index=False)
    print("  Training log saved → checkpoints/training_log.csv")

    with open(NORMALIZER_PATH, "wb") as f:
        pickle.dump(normalizer.state_dict(), f)
    print(f"  Normalizer saved → {NORMALIZER_PATH}")

    # Backtest on held-out test set
    print("\nRunning backtest on test set…")
    test_env = build_env(test_df, sentiment_df=merged_sentiment_df)
    agent.load("checkpoints/best_agent.pt")
    agent_metrics = backtest(agent, test_env, normalizer=normalizer)

    print("\nComputing baselines…")
    comparison = {"SAC Agent": agent_metrics}
    comparison.update(_collect_baselines(test_df, train_df))

    print("\nResults vs. Baselines:")
    print_comparison(comparison)

    try:
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_training_curves, plot_backtest_comparison, plot_weight_heatmap
        os.makedirs("plots", exist_ok=True)

        import pandas as _pd_
        log_df = _pd_.read_csv("checkpoints/training_log.csv")
        plot_training_curves(log_df, save_path="plots/training_curves.png")

        from utils.baselines import equal_weight
        _, baseline_values, baseline_dates = equal_weight(
            test_df, PortfolioEnv.DJ30_TICKERS, initial_capital=1_000_000.0
        )
        agent_values  = np.array(test_env.history["portfolio_value"])
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

    sentiment_df = None
    if args.sentiment:
        sentiment_df = _load_sentiment(args.sentiment_path)
        if sentiment_df is not None:
            from data.sentiment_pipeline import merge_sentiment
            df = merge_sentiment(df, sentiment_df)

    train_df, _, test_df = three_way_split(
        df,
        train_start=TRAIN_START,  train_end=TRAIN_PROPER_END,
        val_start=VAL_START,      val_end=TRAIN_END,
        test_start=TEST_START,    test_end=TEST_END,
    )
    test_env = build_env(test_df, sentiment_df=sentiment_df)
    agent    = build_agent(test_env, config, encoder=args.encoder)
    agent.load(args.checkpoint)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    normalizer = None
    if os.path.exists(NORMALIZER_PATH):
        normalizer = RunningNormalizer(test_env.state_dim, n_skip=test_env.n_assets)
        with open(NORMALIZER_PATH, "rb") as f:
            normalizer.load_state_dict(pickle.load(f))
        print(f"  Loaded normalizer: {NORMALIZER_PATH}")

    agent_metrics = backtest(agent, test_env, normalizer=normalizer)

    print("\nComputing baselines…")
    comparison = {"SAC Agent": agent_metrics}
    comparison.update(_collect_baselines(test_df, train_df))

    print("\nBacktest Results vs. Baselines:")
    print_comparison(comparison)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep RL Portfolio Optimization")
    parser.add_argument("--mode",           choices=["train", "tune", "backtest"], default="train")
    parser.add_argument("--episodes",       type=int, default=100)
    parser.add_argument("--tune-samples",   type=int, default=50)
    parser.add_argument("--config",         type=str, default=None,
                        help="Path to JSON config (from Ray Tune or manual)")
    parser.add_argument("--checkpoint",     type=str, default="checkpoints/best_agent.pt")
    parser.add_argument("--encoder",        choices=["mlp", "transformer"], default="mlp",
                        help="State encoder: flat MLP (default) or AssetTransformerEncoder")
    parser.add_argument("--sentiment",      action="store_true",
                        help="Include precomputed FinBERT sentiment (state_dim 180→210)")
    parser.add_argument("--sentiment-path", type=str, default="data/sentiment_scores.parquet",
                        dest="sentiment_path",
                        help="Path to precomputed sentiment parquet")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(json.load(f))
        print(f"Loaded config from {args.config}")

    if args.mode == "train":
        mode_train(args, config)
    elif args.mode == "tune":
        best = mode_tune(args)
        config.update(best)
        mode_train(args, config)
    elif args.mode == "backtest":
        mode_backtest(args, config)


if __name__ == "__main__":
    main()
