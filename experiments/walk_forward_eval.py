#!/usr/bin/env python3
"""
experiments/walk_forward_eval.py — Phase 3 Multi-Regime Walk-Forward Harness
============================================================================
Wires ``utils/walk_forward.py`` into a proper, honest evaluation instrument —
the walk-forward analogue of ``experiments/multi_seed.py``. It answers: **is the
(now leak-free) SAC agent robust across market regimes, or only on the single
2023–25 window?**

For each seed it runs an expanding-window walk-forward over the full
chronological span (≈2019-04 → 2025-01) so folds sweep COVID (2020), the 2022
bear, and the 2023–24 recovery. For each fold it:

  • recomputes the agent's metrics **net of cost** from the value path (Task A);
  • builds an **equal-weight** baseline over the *same* fold test window;
  • runs a **Jobson–Korkie–Memmel** Sharpe-difference test (agent − EW);
  • records turnover / HHI / active-share.

Results aggregate two ways: **per-regime** (mean ± std + bootstrap 95% CI across
folds/seeds in each regime) and **overall** (pooled). Significance keeps the
Phase-1 honesty convention — per-(seed,fold) JK is the PRIMARY statement; nothing
is averaged in a way that shrinks the standard error.

Resolves the regime part of I-2.

Usage
-----
    # smoke (tiny, fast)
    python experiments/walk_forward_eval.py --seeds 0 --folds 3 --test-months 6 \
        --min-train-months 12 --episodes 5 --warmup 200 --config tuning/best_config.json

    # overnight-able multi-seed run
    python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 \
        --min-train-months 12 --episodes 200 --config tuning/best_config.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAIN_START, TEST_END, UNIVERSE
from environment.portfolio_env import PortfolioEnv
from agent.sac import SACAgent
from utils.seeding import set_global_seed
from utils.run_meta import write_run_meta
from utils.normalizer import RunningNormalizer
from utils.walk_forward import walk_forward
from utils.baselines import equal_weight
from utils import diagnostics as diag
from utils import significance as sig
from utils.regimes import regime_for, REGIME_ORDER

from main import DEFAULT_CONFIG, _load_data


# Row keys that are scalar metrics (everything aggregated / written to CSV).
METRIC_KEYS = [
    "sharpe", "gross_sharpe", "sortino", "calmar", "max_drawdown",
    "total_return", "ann_volatility", "win_rate",
    "mean_turnover", "mean_hhi", "mean_active_share",
    "ew_sharpe", "jk_sharpe_diff_annual",
]


# ── factories ────────────────────────────────────────────────────────────────────

def make_factories(config: dict, encoder: str):
    """Return (agent_factory, normalizer_factory) bound to a tuned config."""
    def agent_factory(state_dim, action_dim):
        return SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
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

    def normalizer_factory(state_dim, n_skip):
        return RunningNormalizer(state_dim, n_skip=n_skip)

    return agent_factory, normalizer_factory


# ── per-fold equal-weight baseline + significance ───────────────────────────────

def _equal_weight_returns(df_full: pd.DataFrame, test_start: str, test_end: str) -> pd.Series:
    """Net-of-cost equal-weight daily returns over a fold's test window."""
    sl = df_full[(df_full["date"] >= test_start) & (df_full["date"] <= test_end)].copy()
    _, values, dates = equal_weight(sl, list(UNIVERSE))
    values = np.asarray(values, dtype=float)
    rets = np.diff(values) / (values[:-1] + 1e-12)
    idx = pd.to_datetime(dates)[1:1 + len(rets)]
    return pd.Series(rets, index=idx, name="equal_weight")


def _fold_significance(agent_rets: np.ndarray, test_dates: list, ew_ret: pd.Series) -> dict:
    """Align agent (net) vs equal-weight (net) and run JK–Memmel on the pair."""
    agent_ser = pd.Series(agent_rets, index=pd.to_datetime(test_dates), name="agent")
    common = agent_ser.index.intersection(ew_ret.index)
    if common.size < 3:
        return {"jk_sharpe_diff_annual": float("nan"), "jk_z": float("nan"),
                "jk_p": float("nan"), "jk_significant": False,
                "ew_sharpe": float("nan"), "n_aligned": int(common.size)}
    a = agent_ser.reindex(common).values
    b = ew_ret.reindex(common).values
    try:
        jk = sig.jobson_korkie_memmel(a, b)
        return {
            "jk_sharpe_diff_annual": jk["sharpe_diff_annual"],
            "jk_z": jk["z_stat"],
            "jk_p": jk["p_value"],
            "jk_significant": bool(jk["p_value"] < 0.05),
            "ew_sharpe": jk["sharpe_b_annual"],
            "agent_sharpe_aligned": jk["sharpe_a_annual"],
            "n_aligned": int(common.size),
        }
    except Exception as e:
        return {"jk_sharpe_diff_annual": float("nan"), "jk_z": float("nan"),
                "jk_p": float("nan"), "jk_significant": False,
                "ew_sharpe": float("nan"), "n_aligned": int(common.size),
                "jk_error": str(e)}


# ── aggregation ──────────────────────────────────────────────────────────────────

def _summarise_rows(rows: list, n_boot: int) -> dict:
    """Aggregate a list of fold rows into {metric: stats} + count summaries."""
    agg = diag.aggregate_metrics(rows, keys=METRIC_KEYS, n_boot=n_boot, seed=0)
    n = len(rows)
    n_beat = sum(1 for r in rows if r.get("beats_ew"))
    p_vals = [r["jk_p"] for r in rows if np.isfinite(r.get("jk_p", np.nan))]
    diffs = [r["jk_sharpe_diff_annual"] for r in rows
             if np.isfinite(r.get("jk_sharpe_diff_annual", np.nan))]
    n_sig = sum(1 for p in p_vals if p < 0.05)
    agg["_counts"] = {
        "n_folds": n,
        "n_beat_equal_weight": n_beat,
        "frac_beat_equal_weight": (n_beat / n) if n else None,
        "n_jk_tests": len(p_vals),
        "n_significant_5pct": n_sig,
        "frac_significant_5pct": (n_sig / len(p_vals)) if p_vals else None,
        "median_jk_p": float(np.median(p_vals)) if p_vals else None,
        "mean_sharpe_diff_annual": float(np.mean(diffs)) if diffs else None,
        "std_sharpe_diff_annual": float(np.std(diffs, ddof=0)) if diffs else None,
    }
    return agg


# ── plots ────────────────────────────────────────────────────────────────────────

def make_plot(regime_agg: dict, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_walk_forward_regimes
    except Exception as e:
        print(f"  Walk-forward plot skipped (import): {e}")
        return
    try:
        path = os.path.join(out_dir, "walk_forward_regimes.png")
        plot_walk_forward_regimes(regime_agg, save_path=path)
        print(f"  Per-regime figure → {path}")
    except Exception as e:
        print(f"  Walk-forward plot skipped: {e}")


# ── core driver (importable by main.py) ─────────────────────────────────────────

def run_walk_forward_eval(
    seeds, folds, test_months, min_train_months, episodes, warmup,
    config_path="tuning/best_config.json", out_dir="experiments/results",
    encoder="mlp", data_path="data/processed_data.parquet",
    n_boot=10_000, n_trials=50,
) -> dict:
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config.update(json.load(f))
        print(f"Loaded config from {config_path}")

    os.makedirs(out_dir, exist_ok=True)
    agent_factory, normalizer_factory = make_factories(config, encoder)

    print("=" * 70)
    print(f"  PHASE 3 WALK-FORWARD EVALUATION  |  seeds={seeds}  folds={folds}  "
          f"test_months={test_months}  min_train_months={min_train_months}  episodes={episodes}")
    print("=" * 70)

    df = _load_data(data_path)
    df["date"] = pd.to_datetime(df["date"])
    # Full chronological span so folds sweep across regimes.
    df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TEST_END)].copy()

    rows = []
    for seed in seeds:
        print(f"\n── Seed {seed} ─────────────────────────────────────────────────")
        set_global_seed(seed)
        fold_results = walk_forward(
            df, agent_factory, normalizer_factory,
            n_folds=folds, test_months=test_months,
            min_train_months=min_train_months,
            train_episodes=episodes, warmup_steps=warmup,
        )
        for fr in fold_results:
            # Leak invariant (defence in depth; walk_forward also asserts it).
            assert fr["train_end"] < fr["test_start"], "LEAK: train_end >= test_start"
            regime = regime_for(fr["test_start"], fr["test_end"])
            ew_ret = _equal_weight_returns(df, fr["test_start"], fr["test_end"])
            sig_d = _fold_significance(fr["agent_returns"], fr["test_dates"], ew_ret)

            row = {
                "seed": int(seed),
                "fold": fr["fold"],
                "regime": regime,
                "train_end": fr["train_end"],
                "test_start": fr["test_start"],
                "test_end": fr["test_end"],
                "n_test_days": fr["n_test_days"],
                "sharpe": fr["sharpe"],                 # NET (honest)
                "gross_sharpe": fr["gross_sharpe"],
                "sortino": fr["sortino"],
                "calmar": fr["calmar"],
                "max_drawdown": fr["max_drawdown"],
                "total_return": fr["total_return"],
                "ann_volatility": fr["ann_volatility"],
                "win_rate": fr["win_rate"],
                "mean_turnover": fr["mean_turnover"],
                "mean_hhi": fr["mean_hhi"],
                "mean_active_share": fr["mean_active_share"],
                "near_uniform": fr["near_uniform"],
                **sig_d,
            }
            row["beats_ew"] = bool(np.isfinite(row["ew_sharpe"]) and row["sharpe"] > row["ew_sharpe"])
            rows.append(row)

    if not rows:
        raise RuntimeError("Walk-forward produced no folds — check the data span / fold settings.")

    # ── Aggregate: per regime + overall ─────────────────────────────────────────
    regimes_seen = [r for r in REGIME_ORDER if any(row["regime"] == r for row in rows)]
    regime_agg = {}
    for rg in regimes_seen:
        regime_agg[rg] = _summarise_rows([row for row in rows if row["regime"] == rg], n_boot)
    overall_agg = _summarise_rows(rows, n_boot)

    # ── Overall deflated Sharpe across all fold net Sharpes (multiple-testing haircut)
    net_sharpes = np.array([row["sharpe"] for row in rows if np.isfinite(row["sharpe"])])
    dsr = None
    if net_sharpes.size >= 2:
        sr_std = float(net_sharpes.std(ddof=0)) / float(np.sqrt(252))  # per-period proxy
        dsr = sig.deflated_sharpe_ratio(
            float(np.median(net_sharpes)) / float(np.sqrt(252)),
            n_obs=int(np.median([row["n_test_days"] for row in rows])),
            skew=0.0, kurt=3.0, n_trials=max(n_trials, net_sharpes.size),
            sr_trials_std=sr_std or None,
        )

    # ── Persist ──────────────────────────────────────────────────────────────────
    csv_rows = [{k: v for k, v in row.items() if k not in ("agent_returns", "test_dates")}
                for row in rows]
    per_fold_csv = os.path.join(out_dir, "walk_forward_per_fold.csv")
    pd.DataFrame(csv_rows).to_csv(per_fold_csv, index=False)

    regime_json = {
        "regimes": regime_agg,
        "overall": overall_agg,
        "regime_bands_doc": {r: True for r in regimes_seen},
    }
    with open(os.path.join(out_dir, "walk_forward_regime.json"), "w") as f:
        json.dump(regime_json, f, indent=2, default=str)

    significance = {
        "framing": ("PRIMARY = per-(seed,fold) Jobson–Korkie–Memmel (each fold a "
                    "full-length OOS record over its test window); pooled DSR is a "
                    "multiple-testing cross-check. No SE-shrinking averaging."),
        "per_regime": {rg: regime_agg[rg]["_counts"] for rg in regimes_seen},
        "overall": overall_agg["_counts"],
        "deflated_sharpe_ratio": dsr,
    }
    with open(os.path.join(out_dir, "walk_forward_significance.json"), "w") as f:
        json.dump(significance, f, indent=2, default=str)

    write_run_meta(out_dir, seed=seeds[0], config=config, device=str(),
                   mode="walk_forward_eval", seeds=list(seeds), folds=folds,
                   test_months=test_months, min_train_months=min_train_months,
                   episodes=episodes, encoder=encoder)

    make_plot(regime_agg, out_dir)

    # ── Report ───────────────────────────────────────────────────────────────────
    _print_report(rows, regime_agg, overall_agg, dsr, regimes_seen, out_dir)

    return {"rows": csv_rows, "regime_agg": regime_agg, "overall": overall_agg,
            "significance": significance}


def _print_report(rows, regime_agg, overall_agg, dsr, regimes_seen, out_dir):
    print("\n" + "=" * 70)
    print("  PER-REGIME — agent NET Sharpe vs Equal-Weight (mean ± std, 95% CI)")
    print("=" * 70)
    print(f"  {'Regime':<24}{'NET Sharpe':>22}{'EW Sharpe':>11}{'k/N>EW':>9}{'sig':>7}")
    print("  " + "─" * 71)
    for rg in regimes_seen:
        a = regime_agg[rg]
        c = a["_counts"]
        s = a.get("sharpe", {})
        ew = a.get("ew_sharpe", {})
        net = f"{s.get('mean', float('nan')):+.3f}±{s.get('std', float('nan')):.3f}" \
              f" [{s.get('ci_low', float('nan')):+.2f},{s.get('ci_high', float('nan')):+.2f}]"
        ewm = f"{ew.get('mean', float('nan')):+.3f}"
        kn = f"{c['n_beat_equal_weight']}/{c['n_folds']}"
        sg = f"{c['n_significant_5pct']}/{c['n_jk_tests']}"
        print(f"  {rg:<24}{net:>22}{ewm:>11}{kn:>9}{sg:>7}")
    print("  " + "─" * 71)

    oc = overall_agg["_counts"]
    os_ = overall_agg.get("sharpe", {})
    og = overall_agg.get("gross_sharpe", {})
    print(f"\n  OVERALL: NET Sharpe {os_.get('mean', float('nan')):+.3f} ± "
          f"{os_.get('std', float('nan')):.3f}  "
          f"[{os_.get('ci_low', float('nan')):+.3f}, {os_.get('ci_high', float('nan')):+.3f}]  "
          f"(gross {og.get('mean', float('nan')):+.3f})")
    print(f"  OVERALL: beats EW in {oc['n_beat_equal_weight']}/{oc['n_folds']} folds; "
          f"{oc['n_significant_5pct']}/{oc['n_jk_tests']} JK tests significant at 5% "
          f"(median p = {oc['median_jk_p']})")
    print(f"  OVERALL: ΔSharpe(annual) vs EW = {oc['mean_sharpe_diff_annual']} "
          f"± {oc['std_sharpe_diff_annual']}")
    if dsr is not None:
        print(f"  Deflated Sharpe Ratio (multiple-testing haircut): DSR = {dsr['dsr']:.4f}")

    print(f"\nArtifacts written to: {out_dir}/")
    print("  walk_forward_per_fold.csv, walk_forward_regime.json,")
    print("  walk_forward_significance.json, run_meta.json, walk_forward_regimes.png")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase 3 multi-regime walk-forward harness")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--folds", type=int, default=9)
    p.add_argument("--test-months", type=int, default=6, dest="test_months")
    p.add_argument("--min-train-months", type=int, default=12, dest="min_train_months",
                   help="months of data before the first test window (12 ⇒ first test ≈2020-04, captures COVID)")
    p.add_argument("--episodes", type=int, default=200, help="train episodes PER FOLD")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--config", type=str, default="tuning/best_config.json")
    p.add_argument("--encoder", choices=["mlp", "transformer"], default="mlp")
    p.add_argument("--out", type=str, default="experiments/results")
    p.add_argument("--data", type=str, default="data/processed_data.parquet")
    p.add_argument("--bootstrap", type=int, default=10_000, dest="bootstrap")
    p.add_argument("--n-trials", type=int, default=50, dest="n_trials")
    args = p.parse_args()

    run_walk_forward_eval(
        seeds=args.seeds, folds=args.folds, test_months=args.test_months,
        min_train_months=args.min_train_months, episodes=args.episodes,
        warmup=args.warmup, config_path=args.config, out_dir=args.out,
        encoder=args.encoder, data_path=args.data, n_boot=args.bootstrap,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
