# SAC Portfolio Optimization — Results

A Soft Actor-Critic agent (Dirichlet policy on the 24-asset simplex) allocates a
fixed large-cap US equity portfolio. This document reports what the agent
actually does under a reproducible, leakage-free, multi-seed, multi-regime,
cost-inclusive evaluation against a panel of standard benchmarks.

## Headline

**Net of realistic transaction costs, the agent does not beat any standard
benchmark.** Its gross (pre-cost) Sharpe is competitive with the panel, but
turnover cost erases the entire edge. Its best-behaving runs converge toward
equal-weight and cannot beat it; its more active runs do worse. This holds
across every regime, every seed, and every intervention tried.

This is a negative result, established rigorously — which replaces an earlier,
unseeded, leaky backtest that had claimed "+24% Sharpe / +47% return over
equal-weight."

## Evaluation setup

- **Universe:** 24 continuous Dow-30 members, 2018-01 → 2025-01 (no
  survivorship/look-ahead bias). State dim 144.
- **Costs:** 0.1% transaction + 0.1% slippage, applied identically to the agent
  and every baseline; all reported Sharpe/return figures are **net of cost**
  (value-path), with gross shown alongside to isolate the cost drag.
- **Single-window:** 5 seeds × 500 episodes, test 2023-01 → 2025-01.
- **Walk-forward:** 3 seeds × 9 expanding folds × 200 episodes, 2019-04 →
  2025-01, leak-free by construction (`train_end < test_start`), covering
  COVID crash, 2021 bull, 2022 rate-shock bear, and 2023-24 recovery.
- **Significance:** per-seed / per-(seed,fold) Jobson–Korkie–Memmel is primary;
  pooled JK, bootstrap CIs, and the Deflated Sharpe Ratio (multiple-testing
  haircut) are cross-checks.
- **Reproducibility:** single seed → identical logs (`make repro` passes).

## Benchmark panel (net of cost, test window)

| Baseline | Sharpe |
|---|---|
| SPY/QQQ 60/40 | +1.98 |
| SPY Buy & Hold | +1.92 |
| 60/40 SPY/AGG | +1.87 |
| Risk Parity | +1.69 |
| Equal Weight | +1.63 |
| Rolling MVO-LW (max-Sharpe) | +1.61 |
| Max-Sharpe MVO (static) | +1.46 |
| Rolling MVO-LW (min-var) | +1.34 |
| Min Variance (static) | +1.05 |
| Momentum 12-1 | +0.89 |

## Single-window result

| Configuration | net Sharpe | gross | turnover | Δ vs strongest (per-seed sig) |
|---|---|---|---|---|
| Prior agent (pre-fixes) | +0.08 | +1.39 | 0.29 | −1.84 (4/5) |
| Entropy fix, no penalty (λ=0) | +0.84 ± 0.60 | +1.62 | 0.17 | −1.09 (3/5) |
| Entropy fix + turnover penalty (λ=0.1–2.0) | +0.50 to +1.03 | ~1.6 | 0.14–0.24 | −0.9 to −1.4 |

The entropy fix roughly halved turnover and lifted net Sharpe from ~0 to ~+0.8.
The turnover penalty added no distinguishable improvement at full scale — the
λ=0 control lands mid-pack and the λ sweep is inside its own noise. The agent
sits above the two weakest baselines (Momentum, static Min-Variance) but below
equal-weight and every strong benchmark.

## Walk-forward result (multi-regime)

| Regime | agent net Sharpe | equal-weight | strongest baseline |
|---|---|---|---|
| COVID crash/recovery | +1.67 ± 0.37 | +2.39 | +2.88 |
| 2021 bull | +0.22 ± 0.24 | +0.71 | +1.14 |
| 2022 bear (rate shock) | −0.42 ± 1.81 | +0.06 | +0.19 |
| 2023-24 recovery/AI | +0.78 ± 1.78 | +1.92 | +2.38 |
| **Overall (27 folds)** | **+0.59 ± 1.54** (gross +1.43) | — | — |

Beats equal-weight in **0/27 folds** and the strongest per-fold baseline in
**0/27**. Versus the prior agent (overall net −0.37, deficit −2.08 vs strongest,
18/27 significant), the gap narrowed substantially (deficit −1.13, 11/27
significant) — but the sign of the conclusion is unchanged. The entropy fix,
present in the λ=0 control here too, is what produced the improvement.

## Why it's stuck (mechanism)

- **The agent's active bets are value-destroying net of cost.** In every
  single-window run, one seed reliably converges to a near-uniform policy
  (turnover ~0.02, Sharpe ~1.5 ≈ equal-weight), while seeds that learn active,
  higher-turnover policies do worse. The best the agent does is rediscover
  diversification.
- **No hyperparameter breaks this.** A reward-scaling sweep across 1e-5 → 1e-2
  (100× the default either way) leaves net Sharpe in a 0.65–0.91 band, all
  within noise, none approaching equal-weight. The entropy temperature α pins at
  its clamp ceiling (5.0) across *every* reward-scaling and turnover-penalty
  setting — the saturation is stable and independent of the reward signal, so it
  is not a "reward too small" problem.
- **Gross ≈ competitive, net ≈ not.** Gross Sharpe (~+1.4 to +1.6) is in the
  benchmark range; the ~0.6–1.0 cost drag from turnover is the entire gap. The
  binding constraint is the absence of net-of-cost edge, not any training knob.

## Phase 5 changes behind these numbers

- **Entropy fix (the real improvement):** the target entropy was re-derived to a
  Dirichlet-attainable value and the temperature bounded (α ∈ [0.01, 5.0]),
  eliminating the earlier blow-up (α → 9.6) / collapse pathology. This is what
  moved net Sharpe from ~0 to positive.
- **Turnover-penalty reward (no measurable effect):** an explicit, tunable
  Σ|Δw| penalty; sweepable, but the λ sweep is a null at full scale.
- **Reproducibility:** CPU thread pinning + action-space seeding; `make repro`
  now passes (identical seeds → identical logs).

## Limitations and caveats

- Seed-to-seed variance is large (per-seed net Sharpe spans ~0.1 to ~1.5); 3–5
  seeds bound but do not eliminate this. Training is unstable in which basin a
  seed lands in.
- The Deflated Sharpe Ratio stays ≈0 throughout — none of these results would
  survive the multiple-testing haircut as a genuine "discovery."
- The α temperature is saturated at its clamp; the clamp is effectively a fixed
  strong entropy regularizer rather than a tuned quantity.
- Hyperparameters were tuned with a short (15-episode) proxy; the transfer to
  500-episode runs is not validated, and a reward-shape change (turnover
  penalty) argues for re-tuning that was not done here.

## Reproduction

```
conda activate portfolio-rl
make test
make repro
python experiments/multi_seed.py --seeds 0 1 2 3 4 --episodes 500 --config tuning/best_config.json --turnover-penalty 0.0 --out experiments/results/tp_0.0
python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 --min-train-months 12 --episodes 200 --config tuning/best_config.json --turnover-penalty 0.5 --out experiments/results/wf_tp_0.5
python experiments/compare_sweep.py experiments/results/tp_0.0 experiments/results/tp_0.1 experiments/results/tp_0.5 experiments/results/tp_1.0 experiments/results/tp_2.0
```
