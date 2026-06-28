# Phase 1 — Honest Evaluation Harness

Replaces the single point estimate with a statistically defensible evaluation:
multiple seeds, confidence intervals, a significance test against equal-weight,
overfitting-aware Sharpe diagnostics, and policy-behavior diagnostics that
explain what the agent actually does. Resolves the core of **I-2**, plus
**I-5** and **I-11**; diagnoses **I-6**. (Leakage fixes remain Phase 2.)

## What was added

| File | Purpose |
|------|---------|
| `utils/significance.py` | Jobson–Korkie test with the **Memmel** correction; stationary (Politis–Romano) block-bootstrap CI on the Sharpe difference; **Probabilistic** and **Deflated Sharpe Ratio**; CSCV **Probability of Backtest Overfitting**. |
| `utils/diagnostics.py` | Turnover (Σ\|Δw\|), concentration (**HHI** = Σw²), **active share** vs equal-weight; per-seed **bootstrap-CI aggregation**; the **return reconciliation** helper. |
| `experiments/multi_seed.py` | The harness: trains + backtests across N seeds, aggregates mean ± std + 95% CI, runs the significance tests, computes diagnostics, persists artifacts, emits two figures. |
| `utils/plotting.py` | `plot_diagnostics_panel` (one-page: turnover, HHI, active share, IS-vs-OOS equity) and `plot_alpha_entropy` (entropy-collapse trajectory). |
| `agent/sac.py` | `update()` now also returns `policy_entropy` (−E[log π]) so the entropy collapse can be logged and plotted (additive, no behavior change). |
| `test/test_significance.py`, `test/test_diagnostics.py` | 22 analytic unit tests for the new math. |
| `Makefile` | `make evaluate` target. |

## How to run

```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization

# quick smoke (a few minutes): low episodes, 3 seeds
python experiments/multi_seed.py --seeds 0 1 2 --episodes 20 --config tuning/best_config.json

# full headline run (overnight on CPU): 5 seeds, 500 episodes
make evaluate                     # == --seeds 0 1 2 3 4 --episodes 500
# or extend later (Colab Pro): SEEDS="0 1 2 3 4 5 6 7 8 9" make evaluate
```

Artifacts land in `experiments/results/`:
`per_seed_results.csv`, `aggregate_metrics.json`, `significance.json`,
`baselines.json`, `run_meta.json`, `diagnostics_panel.png`,
`alpha_entropy_trajectory.png`.

## Acceptance criteria (from the brief) — how each is met

- **Headline reported as mean ± std and 95% CI across ≥5 seeds** → `aggregate_metrics()` (percentile bootstrap on the mean), printed and saved to `aggregate_metrics.json`.
- **p-value / CI on SAC-minus-equal-weight Sharpe, test named** → `jobson_korkie_memmel()` (named) plus `sharpe_diff_bootstrap_ci()` as a cross-check; both in `significance.json`.
- **One-page diagnostic figure (turnover, HHI, active share, IS-vs-OOS equity)** → `diagnostics_panel.png`.
- **All new code committed, no co-author; run_meta accompanies the batch** → done; `run_meta.json` written into the results dir.
- **Reconcile total_return vs ann_return sign discrepancy** → see below.

## The total_return vs ann_return reconciliation (handover §5)

Two effects combined to make `ann_return` positive while `total_return` was
negative:

1. **Costs were excluded from the agent's return array.** The env deducts
   transaction + slippage costs from `portfolio_value` directly, but
   `info["port_return"]` (which fed the metrics) is *gross* of those costs. So
   the mean daily return used by `ann_return` was biased upward, while the
   equity curve (`total_return`) paid the costs. (Baselines were never affected —
   they already derive returns from the value path.)
2. **Arithmetic vs geometric annualization.** `(1+mean)^252−1` ignores
   volatility drag; the realized path is geometric and sits ~0.5·σ² lower.

`reconcile_returns()` reports all three figures side by side
(`ann_return_arith_gross` = legacy/misleading, `ann_return_arith_net` = costs
included, `ann_return_geom_net` = the honest annualized figure consistent with
`total_return`). The harness uses **net-of-cost, value-path returns** for the
agent's metrics and for the significance test, so the agent is measured on the
same basis as the baselines.

## Notes / assumptions

- `--n-trials` (default 50) feeds the Deflated Sharpe haircut; set it to the
  true number of HPO configurations tried for an accurate deflation.
- Significance uses the **across-seed mean** agent return series for the pooled
  test, with per-seed JK tests reported alongside (median p, fraction
  significant).
- Phase 1 only **measures** the entropy collapse (α and policy entropy
  trajectories); changing the entropy mechanism is Phase 5.
- Out of scope (unchanged): HPO test-set leak (I-3) and survivorship (I-4) →
  Phase 2; new baselines (I-7) → Phase 4.
