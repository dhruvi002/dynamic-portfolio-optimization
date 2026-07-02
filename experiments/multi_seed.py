#!/usr/bin/env python3
"""
experiments/multi_seed.py — Phase 1 Honest Evaluation Harness
=============================================================
Replaces the single point estimate with a statistically defensible evaluation:

  • trains + backtests the SAC agent across N seeds;
  • reports every metric as mean ± std and a bootstrap 95% CI;
  • tests the Sharpe difference vs equal-weight with the Jobson–Korkie test
    (Memmel correction) and a stationary block-bootstrap CI cross-check;
  • computes the Deflated Sharpe Ratio to haircut for the number of trials;
  • computes policy-behavior diagnostics (turnover, HHI, active share) and the
    in-sample vs out-of-sample equity curves;
  • logs the alpha / policy-entropy trajectory (diagnoses the entropy collapse);
  • reconciles the total_return vs ann_return discrepancy;
  • persists per-seed raw results, an aggregate summary, the significance
    results, a run_meta.json, and two diagnostic figures.

Resolves the core of I-2, plus I-5, I-11; diagnoses I-6.

Usage
-----
    # quick smoke (low episodes, few seeds)
    python experiments/multi_seed.py --seeds 0 1 2 --episodes 20 --config tuning/best_config.json

    # full headline run
    python experiments/multi_seed.py --seeds 0 1 2 3 4 --episodes 500 --config tuning/best_config.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (TRAIN_START, TRAIN_PROPER_END, VAL_START,
                    TRAIN_END, TEST_START, TEST_END)
from data.pipeline import three_way_split
from environment.portfolio_env import PortfolioEnv
from utils.seeding import set_global_seed
from utils.run_meta import write_run_meta
from utils.trainer import train, backtest
from utils.normalizer import RunningNormalizer
from utils import diagnostics as diag
from utils import significance as sig

# Reuse the exact env/agent/data/baseline construction from main.py so the
# harness measures the *same* pipeline as the single-seed run.
from main import build_env, build_agent, _load_data, _collect_baselines, _collect_baselines_full


# ── per-date return series alignment ────────────────────────────────────────────

def _agent_return_series(env) -> pd.Series:
    """Net-of-cost daily returns of an agent backtest, indexed by date."""
    values = np.asarray(env.history["portfolio_value"], dtype=float)
    rets = diag.returns_from_values(values)
    dates = pd.to_datetime(list(env.dates))[1:1 + len(rets)]
    return pd.Series(rets, index=dates, name="agent")


def _net_metrics(env) -> dict:
    """
    Compute the agent's headline metrics from the **net-of-cost** value-path
    returns (consistent with the equity curve, total_return, and the JK test),
    and expose the **gross** Sharpe alongside so the transaction-cost impact is
    visible. `backtest()`'s own metrics use the env's gross return array and so
    overstate Sharpe/Sortino/Calmar whenever turnover (hence costs) is high.
    """
    from utils.metrics import compute_all_metrics, compute_sharpe
    pv = np.asarray(env.history["portfolio_value"], dtype=float)
    net = diag.returns_from_values(pv)
    gross = np.asarray(env.history["returns"], dtype=float)   # info['port_return'], gross of cost
    m = compute_all_metrics(net, pv)            # net Sharpe/Sortino/Calmar/vol/win_rate
    m["final_value"] = float(pv[-1])
    m["gross_sharpe"] = compute_sharpe(gross)   # annualized, gross of cost
    return m


def _equal_weight_return_series(test_df) -> pd.Series:
    """Equal-weight baseline daily returns, indexed by date (net of costs)."""
    from utils.baselines import equal_weight
    _, values, dates = equal_weight(test_df, PortfolioEnv.DJ30_TICKERS)
    return _baseline_return_series(values, dates, name="equal_weight")


def _baseline_return_series(values, dates, name: str = "baseline") -> pd.Series:
    """
    Phase 4: generic net-of-cost daily return series for ANY baseline's
    (values, dates) output, indexed by date — mirrors
    `_equal_weight_return_series` but works for the full baseline panel.
    """
    values = np.asarray(values, dtype=float)
    rets = np.diff(values) / (values[:-1] + 1e-12)
    idx = pd.to_datetime(dates)[1:1 + len(rets)]
    return pd.Series(rets, index=idx, name=name)


def _collect_baseline_return_series(baselines_full: dict) -> dict:
    """{name: pd.Series} of net daily returns for every baseline that succeeded."""
    out = {}
    for name, (metrics, values, dates) in baselines_full.items():
        try:
            out[name] = _baseline_return_series(values, dates, name=name)
        except Exception as e:
            print(f"  Baseline '{name}' return-series alignment failed: {e}")
    return out


# ── single-seed run ─────────────────────────────────────────────────────────────

def run_one_seed(seed, train_df, val_df, test_df, config, args, ckpt_dir):
    set_global_seed(seed)

    train_env = build_env(train_df, seed=seed)
    val_env = build_env(val_df, seed=seed)
    agent = build_agent(train_env, config, encoder=args.encoder)
    normalizer = RunningNormalizer(train_env.state_dim, n_skip=train_env.n_assets)

    ckpt_path = os.path.join(ckpt_dir, f"seed_{seed}.pt")
    logs = train(
        agent, train_env,
        n_episodes=args.episodes,
        warmup_steps=args.warmup,
        save_path=ckpt_path,
        normalizer=normalizer,
        val_env=val_env,
        seed=seed,
        log_every=max(args.episodes // 5, 1),
    )
    agent.load(ckpt_path)

    # Persist the per-seed normalizer so metrics can be recomputed offline
    # without retraining.
    import pickle
    with open(os.path.join(ckpt_dir, f"seed_{seed}_normalizer.pkl"), "wb") as f:
        pickle.dump(normalizer.state_dict(), f)

    # Out-of-sample (test) deterministic backtest. We call backtest() to *run*
    # the episode (it populates env.history), then recompute metrics on the
    # net-of-cost value path so Sharpe is consistent with total_return.
    test_env = build_env(test_df)
    backtest(agent, test_env, normalizer=normalizer)
    oos_metrics = _net_metrics(test_env)
    oos_diag = diag.policy_diagnostics(test_env)
    agent_ret = _agent_return_series(test_env)

    # In-sample (train) deterministic backtest — for the IS/OOS reconciliation
    is_env = build_env(train_df)
    backtest(agent, is_env, normalizer=normalizer)
    is_metrics = _net_metrics(is_env)
    is_diag = diag.policy_diagnostics(is_env)

    # Reconcile total_return vs ann_return on the OOS path
    recon = diag.reconcile_returns(
        np.asarray(test_env.history["returns"], dtype=float),
        np.asarray(test_env.history["portfolio_value"], dtype=float),
    )

    # Headline per-seed record (value-path / net-of-cost metrics)
    record = {
        "seed": int(seed),
        "sharpe": oos_metrics["sharpe"],                 # net of cost (honest)
        "gross_sharpe": oos_metrics["gross_sharpe"],     # before cost — gap = cost impact
        "sortino": oos_metrics["sortino"],
        "calmar": oos_metrics["calmar"],
        "max_drawdown": oos_metrics["max_drawdown"],
        "total_return": oos_metrics["total_return"],
        "ann_return_geom": recon["ann_return_geom_net"],
        "ann_return_arith": recon["ann_return_arith_gross"],
        "ann_volatility": oos_metrics["ann_volatility"],
        "win_rate": oos_metrics["win_rate"],
        "final_value": oos_metrics["final_value"],
        "mean_turnover": oos_diag["mean_turnover"],
        "mean_hhi": oos_diag["mean_hhi"],
        "mean_active_share": oos_diag["mean_active_share"],
        "near_uniform": oos_diag["near_uniform"],
        "is_sharpe": is_metrics["sharpe"],
        "is_gross_sharpe": is_metrics["gross_sharpe"],
        "is_total_return": is_metrics["total_return"],
        "final_alpha": logs[-1].get("alpha") if logs else None,
        "final_policy_entropy": logs[-1].get("policy_entropy") if logs else None,
    }

    extras = {
        "agent_ret": agent_ret,
        "oos_diag": oos_diag,
        "is_diag": is_diag,
        "logs": logs,
        "recon": recon,
    }
    return record, extras


# ── significance across seeds ───────────────────────────────────────────────────

def _jk_vs_one_baseline(per_seed_extras, base_ret, per_seed_records, args):
    """Jobson–Korkie–Memmel + bootstrap CI for the agent vs a single baseline series."""
    aligned = []
    common = base_ret.index
    for ex in per_seed_extras:
        common = common.intersection(ex["agent_ret"].index)
    if common.size < 3:
        return None
    base_a = base_ret.reindex(common).values
    for ex in per_seed_extras:
        aligned.append(ex["agent_ret"].reindex(common).values)
    aligned = np.vstack(aligned)               # (n_seeds, T)
    agent_mean = aligned.mean(axis=0)          # expected agent return series

    # Per-seed JK tests (distribution of the verdict) — PRIMARY statement.
    per_seed_jk = []
    for i, a in enumerate(aligned):
        try:
            r = sig.jobson_korkie_memmel(a, base_a)
            r["seed"] = per_seed_records[i]["seed"]
            per_seed_jk.append(r)
        except Exception as e:
            per_seed_jk.append({"seed": per_seed_records[i]["seed"], "error": str(e)})

    jk_main = sig.jobson_korkie_memmel(agent_mean, base_a)
    boot = sig.sharpe_diff_bootstrap_ci(
        agent_mean, base_a, n_boot=args.bootstrap, avg_block=args.block, seed=0
    )

    p_vals = [r["p_value"] for r in per_seed_jk if "p_value" in r]
    diffs = [r["sharpe_diff_annual"] for r in per_seed_jk if "sharpe_diff_annual" in r]
    n_sig = sum(1 for p in p_vals if p < 0.05)
    return {
        "n_obs_aligned": int(agent_mean.size),
        # Per-seed is the PRIMARY significance statement: each seed is a genuine,
        # independent track record of full test length.
        "per_seed_jk": per_seed_jk,
        "per_seed_summary": {
            "n_seeds": len(per_seed_jk),
            "mean_sharpe_diff_annual": float(np.mean(diffs)) if diffs else None,
            "std_sharpe_diff_annual": float(np.std(diffs, ddof=0)) if diffs else None,
            "median_p": float(np.median(p_vals)) if p_vals else None,
            "n_significant_5pct": int(n_sig),
            "frac_significant_5pct": float(n_sig / len(p_vals)) if p_vals else None,
        },
        # Pooled test on the cross-seed-AVERAGED series is an OPTIMISTIC
        # cross-check only: averaging shrinks the SE and overstates the z-stat,
        # so it must not be quoted as the headline p-value.
        "jobson_korkie_memmel_pooled_optimistic": jk_main,
        "bootstrap_sharpe_diff": boot,
    }


def run_significance(per_seed_extras, baseline_series: dict, per_seed_records, args):
    """
    Phase 4 (I-7 wiring): per-seed Jobson–Korkie–Memmel of the agent vs EVERY
    baseline in `baseline_series` (not just equal-weight), plus a single
    Deflated Sharpe Ratio on the agent's own expected return series (DSR does
    not depend on the comparison baseline). Returns:
        {
          "vs_baseline": {name: <_jk_vs_one_baseline output>, ...},
          "deflated_sharpe_ratio": {...},
          "strongest_baseline": name,   # highest point-estimate agent-beats-it margin
        }
    "equal_weight" is always included (kept for backward-compat with earlier
    phases' headline framing) if present in `baseline_series`.
    """
    vs_baseline = {}
    for name, base_ret in baseline_series.items():
        r = _jk_vs_one_baseline(per_seed_extras, base_ret, per_seed_records, args)
        if r is not None:
            vs_baseline[name] = r

    # Deflated Sharpe Ratio on the expected agent series (baseline-independent).
    common = None
    for ex in per_seed_extras:
        common = ex["agent_ret"].index if common is None else common.intersection(ex["agent_ret"].index)
    aligned = np.vstack([ex["agent_ret"].reindex(common).values for ex in per_seed_extras])
    agent_mean = aligned.mean(axis=0)
    sr_periodic = sig.periodic_sharpe(agent_mean)
    skew = float(pd.Series(agent_mean).skew())
    kurt = float(pd.Series(agent_mean).kurt() + 3.0)  # pandas gives excess kurtosis
    sr_trials_std = float(np.std([sig.periodic_sharpe(a) for a in aligned], ddof=0))
    dsr = sig.deflated_sharpe_ratio(
        sr_periodic, n_obs=agent_mean.size, skew=skew, kurt=kurt,
        n_trials=args.n_trials, sr_trials_std=sr_trials_std or None,
    )

    # "Strongest baseline" = the one with the highest annualized Sharpe among
    # the baselines actually tested (agent vs the toughest bar, not just EW).
    strongest = None
    best_sr = -np.inf
    for name, r in vs_baseline.items():
        b_sr = r["jobson_korkie_memmel_pooled_optimistic"]["sharpe_b_annual"]
        if np.isfinite(b_sr) and b_sr > best_sr:
            best_sr, strongest = b_sr, name

    return {
        "vs_baseline": vs_baseline,
        "deflated_sharpe_ratio": dsr,
        "strongest_baseline": strongest,
    }


# ── plots ───────────────────────────────────────────────────────────────────────

def make_plots(per_seed_records, per_seed_extras, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_alpha_entropy, plot_diagnostics_panel
    except Exception as e:
        print(f"  Plot generation skipped (import): {e}")
        return

    # Pick the seed closest to the median OOS Sharpe as the representative run.
    sharpes = np.array([r["sharpe"] for r in per_seed_records])
    rep = int(np.argmin(np.abs(sharpes - np.median(sharpes))))
    ex = per_seed_extras[rep]

    try:
        log_df = pd.DataFrame(ex["logs"])
        plot_alpha_entropy(log_df, save_path=os.path.join(out_dir, "alpha_entropy_trajectory.png"))
    except Exception as e:
        print(f"  alpha/entropy plot skipped: {e}")

    try:
        plot_diagnostics_panel(
            ex["oos_diag"], is_diag=ex["is_diag"],
            save_path=os.path.join(out_dir, "diagnostics_panel.png"),
        )
        print(f"  Diagnostics figure → {out_dir}/diagnostics_panel.png (seed {per_seed_records[rep]['seed']})")
    except Exception as e:
        print(f"  diagnostics panel skipped: {e}")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase 1 multi-seed evaluation harness")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--config", type=str, default="tuning/best_config.json")
    p.add_argument("--encoder", choices=["mlp", "transformer"], default="mlp")
    p.add_argument("--out", type=str, default="experiments/results")
    p.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap resamples")
    p.add_argument("--block", type=float, default=10.0, help="avg block length (days) for stationary bootstrap")
    p.add_argument("--n-trials", type=int, default=50, dest="n_trials",
                   help="number of configurations tried (for the Deflated Sharpe haircut)")
    args = p.parse_args()

    from main import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(json.load(f))
        print(f"Loaded config from {args.config}")

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 64)
    print(f"  PHASE 1 MULTI-SEED EVALUATION  |  seeds={args.seeds}  episodes={args.episodes}")
    print("=" * 64)

    df = _load_data("data/processed_data.parquet")
    train_df, val_df, test_df = three_way_split(
        df,
        train_start=TRAIN_START, train_end=TRAIN_PROPER_END,
        val_start=VAL_START, val_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END,
    )

    per_seed_records, per_seed_extras = [], []
    for seed in args.seeds:
        print(f"\n── Seed {seed} ───────────────────────────────────────────────")
        rec, ex = run_one_seed(seed, train_df, val_df, test_df, config, args, ckpt_dir)
        per_seed_records.append(rec)
        per_seed_extras.append(ex)
        print(f"  seed {seed}: OOS Sharpe {rec['sharpe']:.3f} | "
              f"total_return {rec['total_return']:+.2%} | "
              f"turnover {rec['mean_turnover']:.3f} | HHI {rec['mean_hhi']:.4f} | "
              f"active {rec['mean_active_share']:.3f} | near_uniform={rec['near_uniform']}")

    # ── Aggregate ────────────────────────────────────────────────────────────────
    agg_keys = ["sharpe", "gross_sharpe", "sortino", "calmar", "max_drawdown",
                "total_return", "ann_return_geom", "ann_volatility", "win_rate",
                "mean_turnover", "mean_hhi", "mean_active_share", "is_sharpe"]
    agg = diag.aggregate_metrics(per_seed_records, keys=agg_keys,
                                 n_boot=args.bootstrap, seed=0)

    # ── Baselines (deterministic point estimates) — Phase 4 full panel ──────────
    print("\nComputing baselines…")
    baselines_full = _collect_baselines_full(test_df, train_df)
    baselines = {name: m for name, (m, _, _) in baselines_full.items()}

    # ── Significance vs EVERY baseline (Phase 4 wiring of I-7) ──────────────────
    print("\nRunning significance tests (Jobson–Korkie–Memmel + bootstrap + DSR) "
          "vs the full baseline panel…")
    baseline_series = _collect_baseline_return_series(baselines_full)
    ew_ret = baseline_series.get("Equal Weight") or _equal_weight_return_series(test_df)
    significance = run_significance(per_seed_extras, baseline_series, per_seed_records, args)

    # ── Persist ──────────────────────────────────────────────────────────────────
    pd.DataFrame(per_seed_records).to_csv(
        os.path.join(args.out, "per_seed_results.csv"), index=False)
    with open(os.path.join(args.out, "aggregate_metrics.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)
    with open(os.path.join(args.out, "significance.json"), "w") as f:
        json.dump(significance, f, indent=2, default=str)
    with open(os.path.join(args.out, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2, default=str)
    # Per-seed net return series + equal-weight, so metrics/significance can be
    # recomputed offline (no retraining needed).
    np.savez(
        os.path.join(args.out, "per_seed_returns.npz"),
        equal_weight=ew_ret.values,
        ew_dates=ew_ret.index.values.astype("datetime64[D]"),
        **{f"seed_{r['seed']}": ex["agent_ret"].values
           for r, ex in zip(per_seed_records, per_seed_extras)},
    )
    write_run_meta(args.out, seed=args.seeds[0], config=config, device=str(),
                   mode="multi_seed_evaluate", seeds=args.seeds, episodes=args.episodes,
                   encoder=args.encoder, n_trials=args.n_trials)

    make_plots(per_seed_records, per_seed_extras, args.out)

    # ── Report ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  HEADLINE — SAC Agent across {len(args.seeds)} seeds (mean ± std, 95% CI)")
    print("=" * 64)
    print(diag.format_aggregate_table(agg, keys=agg_keys))

    print("\n  Baselines (deterministic, Phase 4 full panel):")
    for name, m in sorted(baselines.items(), key=lambda kv: -kv[1].get("sharpe", float("-inf"))):
        print(f"    {name:<28} Sharpe {m.get('sharpe', float('nan')):+.3f} | "
              f"total_return {m.get('total_return', float('nan')):+.2%}")

    dsr = significance["deflated_sharpe_ratio"]
    strongest = significance["strongest_baseline"]
    print(f"\n  Significance — SAC vs EVERY baseline (headline = vs strongest: {strongest}):")
    print(f"    Deflated Sharpe Ratio (n_trials={dsr['n_trials']}): DSR = {dsr['dsr']:.4f}")
    for name, r in significance["vs_baseline"].items():
        psum = r["per_seed_summary"]
        jk = r["jobson_korkie_memmel_pooled_optimistic"]
        boot = r["bootstrap_sharpe_diff"]
        tag = "  ← HEADLINE (strongest)" if name == strongest else ("  (equal-weight)" if name == "Equal Weight" else "")
        print(f"    vs {name}{tag}")
        print("      PRIMARY (per-seed Jobson–Korkie, Memmel — each seed a full-length record):")
        print(f"        {psum['n_significant_5pct']}/{psum['n_seeds']} seeds significant at 5% "
              f"(median p = {psum['median_p']:.2e})")
        print(f"        ΔSharpe(annual) across seeds = {psum['mean_sharpe_diff_annual']:+.3f} "
              f"± {psum['std_sharpe_diff_annual']:.3f}")
        print("      Cross-checks (optimistic — averaging shrinks SE, do not quote as headline):")
        print(f"        Pooled JK on mean series: ΔSharpe(annual) = {jk['sharpe_diff_annual']:+.3f}, "
              f"z = {jk['z_stat']:+.3f}, p = {jk['p_value']:.2e}")
        print(f"        Bootstrap 95% CI on ΔSharpe(annual): "
              f"[{boot['ci_annual'][0]:+.3f}, {boot['ci_annual'][1]:+.3f}]  "
              f"(excludes 0: {boot['ci_excludes_zero']})")

    # Transaction-cost impact: gross (pre-cost) vs net (post-cost) Sharpe.
    if "gross_sharpe" in agg and "sharpe" in agg:
        g, nsh = agg["gross_sharpe"]["mean"], agg["sharpe"]["mean"]
        print("\n  Transaction-cost impact on Sharpe:")
        print(f"    gross (pre-cost) Sharpe = {g:+.3f}  →  net (post-cost) Sharpe = {nsh:+.3f}  "
              f"(cost drag {g - nsh:+.3f})")
        if g > 0 and nsh < g - 0.3:
            print("    → Costs, driven by high turnover, are the dominant drag on performance.")

    near_uniform_frac = np.mean([r["near_uniform"] for r in per_seed_records])
    print(f"\n  Policy: mean active share = {agg['mean_active_share']['mean']:.3f}, "
          f"mean turnover = {agg['mean_turnover']['mean']:.3f}/step, "
          f"near-uniform in {near_uniform_frac:.0%} of seeds.")
    if near_uniform_frac >= 0.5:
        print("    → The learned policy is approximately equal-weight; this plainly")
        print("      explains why it tracks the equal-weight benchmark so closely.")

    recon0 = per_seed_extras[0]["recon"]
    print("\n  Metric reconciliation (seed {}, OOS):".format(per_seed_records[0]["seed"]))
    print(f"    total_return            = {recon0['total_return']:+.2%}")
    print(f"    ann_return (arith,gross)= {recon0['ann_return_arith_gross']:+.2%}  ← legacy, misleading")
    print(f"    ann_return (arith,net)  = {recon0['ann_return_arith_net']:+.2%}  ← costs included")
    print(f"    ann_return (geom,net)   = {recon0['ann_return_geom_net']:+.2%}  ← honest figure")
    print(f"    daily cost drag         = {recon0['cost_drag_daily']:.2e}")

    print(f"\nArtifacts written to: {args.out}/")
    print("  per_seed_results.csv, aggregate_metrics.json, significance.json,")
    print("  baselines.json, run_meta.json, diagnostics_panel.png, alpha_entropy_trajectory.png")


if __name__ == "__main__":
    main()
