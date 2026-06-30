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
import numpy as np

# NOTE: ray / ray.tune AND hyperopt are imported lazily (inside run_tune(),
# _train_trial(), and _build_search_space()) so this module — and its leak-free
# trial helpers — can be imported for unit testing without a full Ray/HyperOpt
# install (and without tripping hyperopt's pkg_resources/setuptools dependency).
# See test/test_no_leak.py.

# Lazy imports to avoid circular deps
def _build_trial_envs(config: dict):
    """
    Build the train and validation environments for one HPO trial — **never the
    test set**.  (Phase 2, I-3: kill the HPO test-set leak.)

    The old code loaded the *entire* processed dataset (2019 → 2025, INCLUDING the
    2023–2025 test window), built ONE env over the whole series, and reported the
    objective on that same series — so the test set influenced the hyperparameters.
    Here we chronologically split with the canonical `config.py` windows and build:
      • a TRAIN env on  [TRAIN_START, TRAIN_PROPER_END]
      • a VAL   env on  [VAL_START,   TRAIN_END]
    and hard-assert that neither env contains a single test-window row.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import pandas as pd
    from config import (TRAIN_START, TRAIN_PROPER_END, VAL_START,
                        TRAIN_END, TEST_START, TEST_END)
    from data.pipeline import three_way_split
    from environment.portfolio_env import PortfolioEnv

    # Load pre-built dataset (must exist before running tune)
    try:
        df = pd.read_parquet("data/processed_data.parquet")
    except FileNotFoundError:
        # Fallback: synthetic data spanning train+val+test for CI/testing
        df = _synthetic_df()

    train_df, val_df, _test_df = three_way_split(
        df,
        train_start=TRAIN_START, train_end=TRAIN_PROPER_END,
        val_start=VAL_START,     val_end=TRAIN_END,
        test_start=TEST_START,   test_end=TEST_END,
    )

    tc = config.get("tc_rate", 0.001)
    slip = config.get("slip_rate", 0.001)
    train_env = PortfolioEnv(train_df, transaction_cost_rate=tc, slippage_rate=slip)
    val_env   = PortfolioEnv(val_df,   transaction_cost_rate=tc, slippage_rate=slip)

    # ── Hard leakage guard: no env may touch the test window ────────────────────
    test_lo, test_hi = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    train_end_ts = pd.Timestamp(TRAIN_END)
    for name, e in (("train", train_env), ("val", val_env)):
        max_date = pd.Timestamp(max(e.dates))
        if max_date > train_end_ts:
            raise AssertionError(
                f"HPO LEAK GUARD: {name} env max date {max_date.date()} > TRAIN_END "
                f"{train_end_ts.date()} — test-window rows leaked into tuning."
            )
        if any(test_lo <= pd.Timestamp(d) <= test_hi for d in e.dates):
            raise AssertionError(
                f"HPO LEAK GUARD: {name} env contains dates in the test window "
                f"[{TEST_START}, {TEST_END}]."
            )
    return train_env, val_env


def _val_net_sharpe(agent, val_env, normalizer) -> float:
    """
    Deterministic NET-of-cost validation Sharpe — the HPO objective.

    Matches the Phase 1 "net Sharpe" definition (`experiments/multi_seed._net_metrics`)
    so HPO optimizes the exact metric we report: run a deterministic backtest on the
    val env (normalizer frozen), then compute the Sharpe of the value-path returns
    (which include transaction + slippage costs), not the env's gross return array.
    """
    from utils.trainer import backtest
    from utils.diagnostics import returns_from_values
    from utils.metrics import compute_sharpe

    backtest(agent, val_env, normalizer=normalizer)   # populates val_env.history
    pv = np.asarray(val_env.history["portfolio_value"], dtype=float)
    net_returns = returns_from_values(pv)
    return float(compute_sharpe(net_returns))


def _run_one_trial(config: dict) -> float:
    """
    Train one short SAC trial on TRAIN ONLY and return its NET-of-cost VALIDATION
    Sharpe. Shared by the Ray path (`_train_trial`) and the Ray-free path
    (`run_tune_simple`) so both optimize the *identical*, leak-free objective.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agent.sac import SACAgent
    from utils.seeding import set_global_seed
    from utils.trainer import train
    from utils.normalizer import RunningNormalizer

    # ── Reproducible HPO: seed every trial (Phase 2, I-3) ───────────────────────
    set_global_seed(int(config.get("seed", 42)))

    train_env, val_env = _build_trial_envs(config)

    agent = SACAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        gamma=config["gamma"],
        tau=config["tau"],
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        lr_alpha=config["lr_alpha"],
        batch_size=int(config["batch_size"]),
        hidden_sizes=[int(config["hidden_size"])] * 2,
    )

    # Normalizer: skip the first n_assets dims (weights already on the simplex).
    normalizer = RunningNormalizer(train_env.state_dim, n_skip=train_env.n_assets)

    # Short training run on TRAIN ONLY (full training lives in main.py).
    n_episodes = int(config.get("tune_episodes", 15))
    train(
        agent, train_env,
        n_episodes=n_episodes,
        warmup_steps=min(1000, len(train_env.dates) - 1),
        normalizer=normalizer,
        val_env=None,            # objective is computed once below, net of cost
        seed=int(config.get("seed", 42)),
    )

    # Objective: deterministic, net-of-cost VALIDATION Sharpe (no test leakage).
    return _val_net_sharpe(agent, val_env, normalizer)


def _train_trial(config: dict):
    """Ray Tune entry point: run one trial and report the NET val Sharpe."""
    from ray import tune
    val_sharpe = _run_one_trial(config)
    tune.report({"val_sharpe": val_sharpe})


# ─── Ray-free HPO (fallback when Ray/Tune is unavailable in the env) ─────────────

def _sample_config(rng) -> dict:
    """
    Sample one HP config — a numpy mirror of SEARCH_SPACE, so the random-search
    HPO needs neither Ray nor HyperOpt. Reproducible via the passed RNG.
    """
    def loguniform(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return {
        "gamma":       float(rng.uniform(0.95, 0.999)),
        "tau":         loguniform(1e-4, 1e-2),
        "lr_actor":    loguniform(1e-5, 1e-3),
        "lr_critic":   loguniform(1e-5, 1e-3),
        "lr_alpha":    loguniform(1e-5, 1e-3),
        "batch_size":  int(rng.choice([128, 256, 512])),
        "hidden_size": int(rng.choice([128, 256, 512])),
        "tune_episodes": 15,
        "seed": 42,
    }


def run_tune_simple(num_samples: int = 50, search_seed: int = 0) -> dict:
    """
    Ray-free leak-free HPO: sequential random search over the same space, scoring
    each trial by the deterministic NET-of-cost VALIDATION Sharpe (the identical
    objective the Ray path uses). Produces the same `best_config.json`.

    This exists because Ray 2.9.x is incompatible with current setuptools/pyarrow;
    rather than pin a fragile dependency chain just to tune a tiny CPU model, we
    run the search directly. Methodology is disclosed in PHASE2_NOTES.md.
    """
    print("═" * 60)
    print(f"  HPO — Ray-free random search | {num_samples} trials")
    print("  objective = net-of-cost VALIDATION Sharpe (never the test set)")
    print("═" * 60)
    rng = np.random.default_rng(search_seed)
    best_cfg, best_score = None, -np.inf
    for i in range(1, num_samples + 1):
        cfg = _sample_config(rng)
        try:
            score = _run_one_trial(cfg)
        except Exception as e:  # a bad HP combo shouldn't kill the whole sweep
            print(f"  trial {i:3d}/{num_samples}: FAILED ({type(e).__name__}: {e})")
            continue
        flag = ""
        if score > best_score:
            best_score, best_cfg = score, cfg
            flag = "  ← new best"
        print(f"  trial {i:3d}/{num_samples}: val_sharpe={score:+.4f}{flag}")
    print("─" * 60)
    print(f"  Best val_sharpe = {best_score:+.4f}")
    print(f"  Best config     = {best_cfg}")
    print("═" * 60)
    return best_cfg


def _synthetic_df():
    """
    Minimal synthetic data for CI/testing — spans train+val+test so the
    three-way split and the leakage guard have rows to exercise.
    """
    import pandas as pd
    from config import DOWNLOAD_START, TEST_END
    from environment.portfolio_env import PortfolioEnv

    tickers = PortfolioEnv.UNIVERSE
    dates = pd.date_range(DOWNLOAD_START, TEST_END, freq="B")
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

def _build_search_space() -> dict:
    """HyperOpt search space (built lazily so importing this module needs no hyperopt)."""
    from hyperopt import hp
    return {
        "gamma":       hp.uniform("gamma", 0.95, 0.999),
        "tau":         hp.loguniform("tau", np.log(1e-4), np.log(1e-2)),
        "lr_actor":    hp.loguniform("lr_actor", np.log(1e-5), np.log(1e-3)),
        "lr_critic":   hp.loguniform("lr_critic", np.log(1e-5), np.log(1e-3)),
        "lr_alpha":    hp.loguniform("lr_alpha", np.log(1e-5), np.log(1e-3)),
        "batch_size":  hp.choice("batch_size", [128, 256, 512]),
        "hidden_size": hp.choice("hidden_size", [128, 256, 512]),
        "tune_episodes": 15,  # fixed, short for HPO
        "seed": 42,           # fixed seed → reproducible trials (Phase 2, I-3)
    }

# Ray Tune metric the scheduler/search optimize: the deterministic, net-of-cost
# VALIDATION Sharpe (Phase 2). Never the test set.
TUNE_METRIC = "val_sharpe"


def run_tune(
    num_samples: int = 50,
    max_concurrent: int = 4,
    cpus_per_trial: float = 1.0,
    gpus_per_trial: float = 0.0,
    storage_path: str = None,
):
    """Launch the HPO sweep — Ray Tune if available, else Ray-free random search."""
    import os

    # Ray 2.9.x is fragile against current setuptools/pyarrow; if it (or HyperOpt)
    # can't import, fall back to the leak-free random search rather than fail.
    try:
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.hyperopt import HyperOptSearch
    except Exception as e:
        print(f"  [tune] Ray/Tune unavailable ({type(e).__name__}: {e}).")
        print("  [tune] Falling back to Ray-free random-search HPO (same objective).")
        return run_tune_simple(num_samples=num_samples)

    if storage_path is None:
        storage_path = os.path.abspath("ray_results")
    os.makedirs(storage_path, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    scheduler = ASHAScheduler(
        metric=TUNE_METRIC,
        mode="max",
        max_t=20,           # max epochs per trial before pruning
        grace_period=3,     # min epochs before pruning
        reduction_factor=3,
    )

    search_alg = HyperOptSearch(
        space=_build_search_space(),
        metric=TUNE_METRIC,
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
    best = results.get_best_result(metric=TUNE_METRIC, mode="max")

    print("\n" + "═" * 60)
    print("  Best trial (objective = net-of-cost VALIDATION Sharpe, no leak):")
    print(f"  Val Sharpe:  {best.metrics[TUNE_METRIC]:.4f}")
    print(f"  Config:      {best.config}")
    print("═" * 60 + "\n")

    return best.config


if __name__ == "__main__":
    best_config = run_tune(num_samples=50)
    import json
    with open("tuning/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print("Saved best config → tuning/best_config.json")
