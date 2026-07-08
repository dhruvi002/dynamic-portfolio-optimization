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
from utils.seeding import set_global_seed
from utils.run_meta import write_run_meta
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


def build_env(df: pd.DataFrame, sentiment_df: pd.DataFrame = None,
              seed: int = None, turnover_penalty: float = 0.0,
              reward_scaling: float = 1e-4) -> PortfolioEnv:
    return PortfolioEnv(
        df,
        transaction_cost_rate=0.001,
        slippage_rate=0.001,
        initial_capital=1_000_000.0,
        turnover_penalty=turnover_penalty,   # Phase 5 (Task B); 0.0 = no change
        reward_scaling=reward_scaling,        # sweepable; controls critic-vs-entropy balance
        sentiment_df=sentiment_df,
        seed=seed,
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
        # Phase 5 (Task A) entropy controls; config-overridable, safe defaults.
        target_conc=config.get("target_conc", 2.0),
        alpha_init=config.get("alpha_init", 1.0),
        alpha_min=config.get("alpha_min", 0.01),
        alpha_max=config.get("alpha_max", 5.0),
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

def _collect_baselines_full(test_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    """
    Run all baseline strategies and return {name: (metrics_dict, values, dates)}.

    Phase 4 (I-7): the strong baseline set — equal-weight, momentum, static MVO
    (kept for backward-compat / "static MVO" contrast) PLUS the standard
    benchmarks a practitioner would respect: SPY buy-and-hold, the real 60/40
    SPY/AGG balanced portfolio, inverse-vol risk parity, and rolling
    Ledoit-Wolf MVO (min-var + max-Sharpe) — all net of cost, no look-ahead.
    """
    from utils.baselines import (
        equal_weight, spy_qqq, momentum_12_1, min_variance, max_sharpe_mvo,
        spy_buy_and_hold, spy_agg_60_40, risk_parity, rolling_mvo_ledoit_wolf,
    )
    tickers = PortfolioEnv.DJ30_TICKERS
    results = {}

    def _try(name, fn, *a, **kw):
        try:
            results[name] = fn(*a, **kw)
        except Exception as e:
            print(f"  Baseline '{name}' failed: {e}")

    _try("Equal Weight", equal_weight, test_df, tickers)
    _try("SPY/QQQ 60/40", spy_qqq, start=TEST_START, end=TEST_END)
    _try("Momentum 12-1", momentum_12_1, test_df, tickers)
    _try("Min Variance (static)", min_variance, test_df, tickers, train_df=train_df)
    _try("Max Sharpe MVO (static)", max_sharpe_mvo, test_df, tickers, train_df=train_df)
    _try("SPY Buy&Hold", spy_buy_and_hold, start=TEST_START, end=TEST_END)
    _try("60/40 SPY/AGG", spy_agg_60_40, start=TEST_START, end=TEST_END)
    _try("Risk Parity", risk_parity, test_df, tickers, train_df=train_df)
    _try("Rolling MVO-LW (min-var)", rolling_mvo_ledoit_wolf, test_df, tickers,
         kind="min_var", train_df=train_df)
    _try("Rolling MVO-LW (max-Sharpe)", rolling_mvo_ledoit_wolf, test_df, tickers,
         kind="max_sharpe", train_df=train_df)

    return results


def _collect_baselines(test_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    """Backward-compat wrapper: {name: metrics_dict} only (used by print_comparison)."""
    return {name: m for name, (m, _, _) in _collect_baselines_full(test_df, train_df).items()}


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

    train_env = build_env(train_df, sentiment_df=merged_sentiment_df, seed=args.seed,
                          turnover_penalty=config.get("turnover_penalty", 0.0),
                          reward_scaling=config.get("reward_scaling", 1e-4))
    val_env   = build_env(val_df, sentiment_df=merged_sentiment_df, seed=args.seed)
    agent     = build_agent(train_env, config, encoder=args.encoder)
    print(f"  State dim: {train_env.state_dim}  |  Action dim: {train_env.action_dim}")
    print(f"  Device: {agent.device}")

    meta_path = write_run_meta(
        "checkpoints", seed=args.seed, config=config, device=agent.device,
        mode="train", episodes=args.episodes, encoder=args.encoder,
        sentiment=bool(args.sentiment),
    )
    print(f"  Run metadata stamped \u2192 {meta_path}")

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
        seed=args.seed,
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


def mode_walkforward(args):
    """Phase 3: multi-regime walk-forward evaluation (delegates to the harness)."""
    print("=" * 60)
    print("  MODE: WALK-FORWARD (Phase 3, multi-regime)")
    print("=" * 60)
    from experiments.walk_forward_eval import run_walk_forward_eval
    run_walk_forward_eval(
        seeds=args.wf_seeds,
        folds=args.folds,
        test_months=args.test_months,
        min_train_months=args.min_train_months,
        episodes=args.episodes,
        warmup=1000,
        config_path=args.config,
        encoder=args.encoder,
    )


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
    test_env = build_env(test_df, sentiment_df=sentiment_df, seed=args.seed)
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
    parser.add_argument("--mode",           choices=["train", "tune", "backtest", "walkforward"], default="train")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Global RNG seed for reproducible runs")
    parser.add_argument("--episodes",       type=int, default=100)
    parser.add_argument("--wf-seeds",       type=int, nargs="+", default=[0, 1, 2], dest="wf_seeds",
                        help="Seeds for walk-forward mode")
    parser.add_argument("--folds",          type=int, default=9, help="Walk-forward folds")
    parser.add_argument("--test-months",    type=int, default=6, dest="test_months",
                        help="Walk-forward test window length (months)")
    parser.add_argument("--min-train-months", type=int, default=12, dest="min_train_months",
                        help="Months of data before the first walk-forward test window")
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

    set_global_seed(args.seed)
    print(f"Global seed set: {args.seed}")

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
    elif args.mode == "walkforward":
        mode_walkforward(args)


if __name__ == "__main__":
    main()
