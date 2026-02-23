"""
Ray Tune Hyperparameter Optimization
=====================================
Searches for the optimal SAC configuration across 50+ parallel runs.

Key hyperparameters searched:
  - entropy_alpha (init):  controls explore/exploit trade-off
  - lr_actor / lr_critic:  learning rate balance
  - gamma:                 discount horizon
  - tau:                   target network update rate
  - hidden_sizes:          network capacity
  - batch_size:            sample efficiency

Scheduler: ASHA (Asynchronous Successive Halving) — early-stops bad trials.
Search:    HyperOpt (Tree Parzen Estimator) for sample-efficient Bayesian search.

Usage:
    python -m tuning.tune_runner
"""

import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
import numpy as np

# Lazy imports to avoid circular deps
def _train_trial(config: dict):
    """Single Ray Tune trial: train SAC and report Sharpe ratio."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import pandas as pd
    from agent.sac import SACAgent
    from environment.portfolio_env import PortfolioEnv
    from utils.metrics import compute_sharpe

    # Load pre-built dataset (must exist before running tune)
    try:
        df = pd.read_parquet("data/processed_data.parquet")
    except FileNotFoundError:
        # Fallback: generate synthetic data for testing
        df = _synthetic_df()

    env = PortfolioEnv(
        df,
        transaction_cost_rate=config.get("tc_rate", 0.001),
        slippage_rate=config.get("slip_rate", 0.001),
    )

    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=config["gamma"],
        tau=config["tau"],
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        lr_alpha=config["lr_alpha"],
        batch_size=int(config["batch_size"]),
        hidden_sizes=[int(config["hidden_size"])] * 2,
    )

    # Warm-up: fill buffer with random actions
    state, _ = env.reset()
    for _ in range(min(1000, len(env.dates) - 1)):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        state = next_state if not done else env.reset()[0]

    # Training loop (shortened for HPO — full training in main.py)
    n_episodes = config.get("tune_episodes", 10)
    portfolio_returns = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_returns = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            ep_returns.append(info["port_return"])
            state = next_state

        portfolio_returns.extend(ep_returns)

    sharpe = compute_sharpe(np.array(portfolio_returns))

    tune.report({"sharpe": sharpe, "mean_return": float(np.mean(portfolio_returns))})


def _synthetic_df():
    """Generate minimal synthetic data for CI/testing."""
    import pandas as pd
    from environment.portfolio_env import PortfolioEnv

    tickers = PortfolioEnv.DJ30_TICKERS
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    rows = []
    prices = np.ones(len(tickers)) * 100.0
    for date in dates:
        prices = prices * np.exp(np.random.randn(len(tickers)) * 0.01)
        for i, tic in enumerate(tickers):
            rows.append({
                "date": date,
                "tic": tic,
                "open": prices[i],
                "high": prices[i] * 1.005,
                "low": prices[i] * 0.995,
                "close": prices[i],
                "volume": 1e6,
                "macd": 0.0, "rsi_30": 50.0, "cci_30": 0.0, "dx_30": 25.0,
            })
    return pd.DataFrame(rows)


# ─── Search space ──────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "gamma":       hp.uniform("gamma", 0.95, 0.999),
    "tau":         hp.loguniform("tau", np.log(1e-4), np.log(1e-2)),
    "lr_actor":    hp.loguniform("lr_actor", np.log(1e-5), np.log(1e-3)),
    "lr_critic":   hp.loguniform("lr_critic", np.log(1e-5), np.log(1e-3)),
    "lr_alpha":    hp.loguniform("lr_alpha", np.log(1e-5), np.log(1e-3)),
    "batch_size":  hp.choice("batch_size", [128, 256, 512]),
    "hidden_size": hp.choice("hidden_size", [128, 256, 512]),
    "tune_episodes": 15,  # fixed, short for HPO
}


def run_tune(
    num_samples: int = 50,
    max_concurrent: int = 4,
    cpus_per_trial: float = 1.0,
    gpus_per_trial: float = 0.0,
    storage_path: str = None,
):
    """Launch Ray Tune HPO sweep."""
    import os
    if storage_path is None:
        storage_path = os.path.abspath("ray_results")
    os.makedirs(storage_path, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    scheduler = ASHAScheduler(
        metric="sharpe",
        mode="max",
        max_t=20,           # max epochs per trial before pruning
        grace_period=3,     # min epochs before pruning
        reduction_factor=3,
    )

    search_alg = HyperOptSearch(
        space=SEARCH_SPACE,
        metric="sharpe",
        mode="max",
        n_initial_points=10,   # random exploration before TPE kicks in
    )

    tuner = tune.Tuner(
        tune.with_resources(
            _train_trial,
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=tune.RunConfig(
            storage_path=storage_path,
            name="sac_portfolio_hpo",
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="sharpe", mode="max")

    print("\n" + "═" * 60)
    print("  Best trial:")
    print(f"  Sharpe:  {best.metrics['sharpe']:.4f}")
    print(f"  Config:  {best.config}")
    print("═" * 60 + "\n")

    return best.config


if __name__ == "__main__":
    best_config = run_tune(num_samples=50)
    import json
    with open("tuning/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print("Saved best config → tuning/best_config.json")
