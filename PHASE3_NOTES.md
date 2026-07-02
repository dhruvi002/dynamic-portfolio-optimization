# Phase 3 — Multi-Regime Walk-Forward Evaluation (design notes)

Companion to `PHASE2_HANDOVER.md` §9. Resolves the **regime part of I-2**: shows
whether the (leak-free) SAC agent is robust *across market regimes*, not only on
the single 2023–25 test window. Net-of-cost throughout, per-regime CIs, per-fold
significance vs equal-weight.

## What changed

### Task A — `utils/walk_forward.py` now reports NET-of-cost metrics
The old `walk_forward()` reported `trainer.backtest()`'s **gross** Sharpe/Sortino/
Calmar — the exact bug Phase 1 fixed for the single-window harness. Each fold now:
- recomputes Sharpe/Sortino/Calmar/total_return from the **net-of-cost value path**
  (`diagnostics.returns_from_values`), mirroring `multi_seed._net_metrics`, and
  exposes `gross_sharpe` alongside so the transaction-cost drag is visible;
- attaches per-fold **turnover / HHI / active-share** (`policy_diagnostics`);
- attaches the raw net daily return series (`agent_returns` + aligned `test_dates`)
  so the harness can run a per-fold significance test;
- asserts `train_end < test_start` (leak-free by construction, now enforced).

Fold scheduling is factored into a pure helper `_fold_windows(...)` so the
chronology / non-overlap / expanding-train / clean-stop properties are unit-tested
without any training.

### Task B — `experiments/walk_forward_eval.py` (the harness)
Mirrors `experiments/multi_seed.py`. For each seed it `set_global_seed(seed)` and
runs `walk_forward(...)` over the **full** span `[TRAIN_START, TEST_END]` (built
from `tuning/best_config.json`, encoder `mlp`). For each fold it builds an
**equal-weight baseline over the same fold test window**, runs **Jobson–Korkie–
Memmel** (agent − EW), and records turnover/HHI/active-share. Aggregates two ways:
**per-regime** and **overall**, each with mean ± std + bootstrap 95% CI
(`diagnostics.aggregate_metrics` / `bootstrap_ci`). Persists:
`walk_forward_per_fold.csv`, `walk_forward_regime.json`,
`walk_forward_significance.json`, `run_meta.json`, and `walk_forward_regimes.png`.

### Task C — regime labelling (`utils/regimes.py`)
Each fold is assigned to a regime by its test-window **midpoint**. Documented,
fixed cutoffs:

| Regime | Inclusive dates |
|---|---|
| pre-COVID 2019 | 2019-01-01 → 2020-01-31 |
| COVID crash/recovery | 2020-02-01 → 2020-12-31 |
| 2021 bull | 2021-01-01 → 2021-12-31 |
| 2022 bear (rate shock) | 2022-01-01 → 2022-12-31 |
| 2023-24 recovery/AI | 2023-01-01 → 2025-12-31 |

With the defaults (`min_train_months=12`, `test_months=6`, `folds=9`) the folds
sweep all four active regimes with ≥2 folds each:

```
fold1 2020-04→2020-09  COVID        fold6 2022-10→2023-03  2022 bear
fold2 2020-10→2021-03  COVID        fold7 2023-04→2023-09  recovery/AI
fold3 2021-04→2021-09  2021 bull    fold8 2023-10→2024-03  recovery/AI
fold4 2021-10→2022-03  2021 bull    fold9 2024-04→2024-09  recovery/AI
fold5 2022-04→2022-09  2022 bear
```

`min_train_months=12` (first test ≈ 2020-04) is the deliberate choice that buys
COVID-regime coverage; raise it for more training data per fold at the cost of the
earliest regime.

### Task D — `test/test_walk_forward.py`
Synthetic-data only (no downloads, no Ray, no SAC training): fold chronology /
non-overlap / expanding-train / `train_end < test_start`; clean stop past the data
window; the **net** metric path differs from the gross `backtest()` Sharpe when
turnover > 0 and is derived from the value path; regime labelling.

### Wiring
`make walkforward` (overridable `WF_SEEDS FOLDS TEST_MONTHS MIN_TRAIN_MONTHS
WF_EPISODES`) and `python main.py --mode walkforward`.

## Significance framing (unchanged honesty convention)
**PRIMARY = per-(seed,fold) Jobson–Korkie–Memmel** — each fold is a full-length
OOS record over its test window. The overall **Deflated Sharpe Ratio** is a
multiple-testing cross-check. No SE-shrinking averaging is quoted as a headline.

## How to run
```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization
pip install pytest >/dev/null 2>&1 ; make test
python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 --min-train-months 12 --episodes 200 --config tuning/best_config.json
```
Budget: code is done; the sweep is folds × seeds × episodes — keep `--episodes`
at 150–250 to stay overnight-able on the MacBook Air (CPU).

## Result (3 seeds × 9 folds, 200 ep/fold, net of cost)

The Phase-2 single-window verdict **generalizes across regimes**: the agent loses
to equal-weight net in **every** fold.

| Regime | Agent NET Sharpe (mean ± std, 95% CI) | EW Sharpe | k/N > EW | JK sig @5% |
|---|---|---|---|---|
| COVID crash/recovery | +1.294 ± 0.422 [+0.96, +1.63] | +2.387 | 0/6 | 5/6 |
| 2021 bull | −0.972 ± 0.403 [−1.28, −0.64] | +0.708 | 0/6 | 6/6 |
| 2022 bear (rate shock) | −1.410 ± 1.465 [−2.44, −0.25] | +0.063 | 0/6 | 6/6 |
| 2023-24 recovery/AI | −0.590 ± 1.320 [−1.45, +0.26] | +1.920 | 0/9 | 8/9 |

**Overall:** NET Sharpe **−0.438 ± 1.443** [−0.982, +0.103] vs **gross +1.145** —
the ~1.58 gross→net collapse is the transaction-cost drag from ~0.4/step turnover,
the same mechanism identified in Phase 2 §6. Beats EW in **0/27 folds**; **25/27**
per-fold JK tests significant at 5% (median p ≈ 1.7e-4); ΔSharpe(annual) vs EW =
**−1.780 ± 0.806**. **Deflated Sharpe Ratio = 0.0045** ≈ 0 (matches the single-window
DSR 0.004) — does not survive the multiple-testing haircut.

Nuance worth disclosing: the agent's *only* positive-Sharpe regime is COVID
(+1.29), but even there equal-weight is higher (+2.39), so it is not "competitive"
anywhere net of cost. Turnover, not signal, is the binding constraint — the Phase-5
cost-aware-reward / turnover-penalty work is what could plausibly move this.

## Out of scope (per handover §9.7)
Entropy/reward fixes (I-6, I-10 → Phase 5); new baselines (I-7 → Phase 4);
README/doc rewrite & state-dim/FinRL fixes (I-9 → Phase 6); longer-proxy HPO
re-tune (I-8 → Phase 5/6).
