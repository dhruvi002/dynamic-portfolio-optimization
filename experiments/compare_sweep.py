#!/usr/bin/env python3
"""
experiments/compare_sweep.py — side-by-side comparison of turnover-penalty runs.

Phase 5 (Task B) helper. Reads the per-seed results + significance JSON that
`multi_seed.py` writes into each --out directory and prints one row per run so a
λ_turnover sweep can be read at a glance:

  - NET Sharpe (mean ± std across seeds) — the honest, cost-inclusive headline
  - GROSS Sharpe (mean) and the GROSS→NET gap (the cost drag Phase 5 targets)
  - mean turnover / step (should fall as λ_turnover rises — the mechanism check)
  - α range across seeds (final_alpha min→max) — Task A boundedness check
  - Δ Sharpe (annual) vs the strongest baseline + how many seeds are significant

Usage:
    python experiments/compare_sweep.py                       # auto-globs scan/tp dirs
    python experiments/compare_sweep.py experiments/results/scan_tp_0.1 \
        experiments/results/scan_tp_0.5 experiments/results/scan_tp_1.0

Nothing here retrains or claims a result — it only tabulates finished runs. Read
the short-scan numbers for DIRECTION and α-boundedness, not as final figures.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd


def _label(d: str) -> str:
    """Best-effort label from the stamped config (λ and/or reward_scaling), else dir name."""
    for meta in ("run_meta.json", "aggregate_metrics.json"):
        p = os.path.join(d, meta)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    blob = json.load(f)
                cfg = blob.get("config", blob)
                if isinstance(cfg, dict) and ("turnover_penalty" in cfg or "reward_scaling" in cfg):
                    parts = []
                    if "turnover_penalty" in cfg:
                        parts.append(f"λ={float(cfg['turnover_penalty']):g}")
                    if "reward_scaling" in cfg:
                        parts.append(f"rs={float(cfg['reward_scaling']):g}")
                    return " ".join(parts)
            except Exception:
                pass
    return os.path.basename(os.path.normpath(d))


def _strongest(d: str):
    """(name, mean ΔSharpe annual, n_sig, n_seeds) vs the strongest baseline."""
    p = os.path.join(d, "significance.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            sig = json.load(f)
        name = sig.get("strongest_baseline")
        summ = sig["vs_baseline"][name]["per_seed_summary"]
        return (name, summ.get("mean_sharpe_diff_annual"),
                summ.get("n_significant_5pct"), summ.get("n_seeds"))
    except Exception:
        return None


def summarize(d: str) -> dict | None:
    csv = os.path.join(d, "per_seed_results.csv")
    if not os.path.exists(csv):
        print(f"  [skip] no per_seed_results.csv in {d}")
        return None
    df = pd.read_csv(csv)

    def col(name):
        return df[name] if name in df.columns else pd.Series(dtype=float)

    net = col("sharpe")
    gross = col("gross_sharpe")
    alpha = col("final_alpha").dropna()
    row = {
        "run": _label(d),
        "seeds": len(df),
        "net_sharpe_mean": float(net.mean()) if len(net) else math.nan,
        "net_sharpe_std": float(net.std(ddof=0)) if len(net) else math.nan,
        "gross_sharpe_mean": float(gross.mean()) if len(gross) else math.nan,
        "mean_turnover": float(col("mean_turnover").mean()) if "mean_turnover" in df else math.nan,
        "active_share": float(col("mean_active_share").mean()) if "mean_active_share" in df else math.nan,
        "alpha_min": float(alpha.min()) if len(alpha) else math.nan,
        "alpha_max": float(alpha.max()) if len(alpha) else math.nan,
    }
    row["gap"] = row["gross_sharpe_mean"] - row["net_sharpe_mean"]
    row["strongest"] = _strongest(d)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*",
                    help="result dirs; default globs experiments/results/{scan_,}tp_*")
    args = ap.parse_args()

    dirs = args.dirs or sorted(
        set(glob.glob("experiments/results/scan_tp_*")
            + glob.glob("experiments/results/tp_*")))
    if not dirs:
        print("No result dirs found. Pass them explicitly, e.g.:\n"
              "  python experiments/compare_sweep.py experiments/results/scan_tp_0.5")
        return

    rows = [r for r in (summarize(d) for d in dirs) if r]
    if not rows:
        return
    rows.sort(key=lambda r: (math.isnan(r["net_sharpe_mean"]), -r["net_sharpe_mean"]))

    hdr = (f"{'run':<16}{'seeds':>6}{'net Sharpe':>18}{'gross':>9}"
           f"{'gap':>8}{'turnover':>10}{'act.sh':>8}{'alpha[min,max]':>18}")
    print("\n" + "=" * len(hdr))
    print("  TURNOVER-PENALTY SWEEP  (net-of-cost headline)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        net = f"{r['net_sharpe_mean']:+.3f} ± {r['net_sharpe_std']:.3f}"
        arng = f"[{r['alpha_min']:.3f}, {r['alpha_max']:.3f}]"
        print(f"{r['run']:<16}{r['seeds']:>6}{net:>18}"
              f"{r['gross_sharpe_mean']:>9.3f}{r['gap']:>8.3f}"
              f"{r['mean_turnover']:>10.3f}{r['active_share']:>8.3f}{arng:>18}")
    print("-" * len(hdr))

    print("\nvs strongest baseline (per-seed JK, the Phase-4 headline bar):")
    for r in rows:
        s = r["strongest"]
        if s is None:
            print(f"  {r['run']:<16} (no significance.json)")
            continue
        name, dsh, nsig, nse = s
        dsh_s = f"{dsh:+.3f}" if dsh is not None else "n/a"
        print(f"  {r['run']:<16} vs {str(name):<26} "
              f"ΔSharpe(ann)={dsh_s}  sig {nsig}/{nse} seeds")

    print("\nReading guide:")
    print("  • gap should SHRINK as λ rises if the penalty is working (less")
    print("    turnover eaten by cost); net Sharpe rising toward the panel is the goal.")
    print("  • alpha[min,max] must stay inside the clamp band [0.01, 5.0] (Task A).")
    print("  • short-scan numbers are noisy — pick DIRECTION, then do the full run.")


if __name__ == "__main__":
    main()
