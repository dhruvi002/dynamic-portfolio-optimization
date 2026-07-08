# Phase 4 — Strong Baseline Overhaul (design notes)

Companion to `PHASE3_HANDOVER.md` §8. Resolves **I-7**: replaces the weak/strawman
baseline set (equal-weight, a mislabeled "60/40" that was actually SPY/QQQ, and
static single-estimate MVO) with a strong, standard panel a practitioner would
actually respect, and re-runs both the single-window and walk-forward harnesses
against the full panel.

## What changed

### Task A — `utils/baselines.py`: four new baselines
- **`spy_buy_and_hold(start, end, ...)`** — single-asset SPY, one entry cost at
  t0 (Σ|Δw|=1, same cost formula as everything else), then held with zero
  further cost. The canonical market benchmark, previously missing entirely.
- **`spy_agg_60_40(start, end, ...)`** — 60% SPY / 40% AGG, monthly rebalance.
  The *real* 60/40 balanced benchmark (`spy_qqq` — kept for backward
  compatibility — is two tech-heavy equity ETFs, not a balanced portfolio).
- **`risk_parity(df, tickers, train_df=None, lookback=252, ...)`** —
  inverse-volatility weights, re-estimated every rebalance from a trailing
  252-day window of data with `date <= rebalance_date` only (no look-ahead).
- **`rolling_mvo_ledoit_wolf(df, tickers, kind="min_var"|"max_sharpe", ...)`**
  — "MVO done properly": Ledoit-Wolf shrunk covariance (`sklearn.covariance.
  LedoitWolf`, lazy-imported with a sample-covariance fallback + printed
  warning if scikit-learn is absent), re-estimated every rebalance on a
  trailing 252-day window — replaces the old static single-estimate
  `min_variance`/`max_sharpe_mvo` as the MVO headline (those two are kept as a
  "static MVO" contrast).

All four share the existing `(tc_rate+slip_rate)·Σ|Δw|·value` cost model and the
`(metrics_dict, values, dates)` return signature.

**Perf fix (post-hoc):** the first walk-forward run called `spy_buy_and_hold`/
`spy_agg_60_40` once per fold per seed (54 yfinance calls total) and hit a
multi-hour rate-limit stall on one fold (33h wall-clock on an otherwise ~20min
fold). Fixed by adding `spy_buy_and_hold_from_series`/`spy_agg_60_40_from_series`
plus a memoizing `_yf_download` cache; the walk-forward harness now downloads
SPY/AGG **once** for the full span up front and slices the cached series per
fold instead of re-hitting the network per fold.

### Task B — wired into both harnesses
- **`main._collect_baselines_full`** — runs the full 10-strategy panel (5 old +
  4 new + `spy_qqq` kept), returns `{name: (metrics, values, dates)}`.
  `_collect_baselines` is now a thin backward-compat wrapper returning just the
  metrics dicts (used by `main.py`'s `print_comparison` tables).
- **`experiments/multi_seed.py`** — per-seed Jobson–Korkie–Memmel of the agent
  vs **every** baseline in the panel (previously only equal-weight), plus a
  `"strongest_baseline"` field (highest point-estimate annualized Sharpe among
  the baselines that ran) reported as the headline comparison. Equal-weight's
  comparison is kept alongside for continuity with Phases 1–3.
- **`experiments/walk_forward_eval.py`** — each fold computes every baseline
  over that fold's test window, with estimators fed only
  `train_df_fold = df[date < test_start]` (identical leak-free discipline to
  the agent's own expanding training window). `walk_forward_per_fold.csv` now
  carries, per baseline: `{sanitized_name}_sharpe`, `_jk_diff_annual`, `_jk_p`,
  `_beats`. The per-regime/overall aggregation additionally reports **agent vs
  the strongest baseml this fold** (`best_baseline_name`, `best_baseline_sharpe`,
  `jk_vs_best_*`, `_counts_vs_best`) as the Phase-4 headline table, printed
  alongside the original equal-weight-only table.

### Task C — figure
`utils.plotting.plot_walk_forward_baseline_panel` — grouped bars (agent +
every baseline) per regime with bootstrap 95% CI error bars, companion to the
existing `plot_walk_forward_regimes` (agent vs EW only).

### Task D — `test/test_baselines.py` (30 new tests, 40 total in the file)
Cost-model consistency (hand-computed single-rebalance cases), no-look-ahead
(prefix-invariance: truncating the test window to an earlier end date must not
change the value path up to the shared truncation point — proven for both
`risk_parity` and `rolling_mvo_ledoit_wolf`), well-formed weights (sum-to-one,
non-negative, inverse-vol ordering), Ledoit-Wolf shrinkage (off-diagonal
magnitude vs sample covariance on a noisy synthetic panel) with a graceful
no-sklearn fallback test (simulated via `builtins.__import__` monkeypatching),
and SPY buy-and-hold's one-time entry cost (value-curve ratio to a zero-cost
run is flat over time). yfinance calls are mocked (`monkeypatch.setattr(yf,
"download", ...)`) so the suite needs no network access.

## Significance framing (unchanged honesty convention)
**PRIMARY = per-seed / per-(seed,fold) Jobson–Korkie–Memmel.** Pooled JK on the
cross-seed-averaged series and the bootstrap CI are optimistic cross-checks
only (averaging shrinks the standard error). DSR is the multiple-testing
haircut. Phase 4 adds: the **headline comparison is now vs the strongest
baseline**, not equal-weight — a harder bar, reported alongside the
EW-specific numbers for continuity.

## How to run
```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization
pip install scikit-learn pytest >/dev/null 2>&1 ; make lock ; make test
caffeinate -i python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 --min-train-months 12 --episodes 200 --config tuning/best_config.json
caffeinate -i python experiments/multi_seed.py --seeds 0 1 2 3 4 --episodes 500 --config tuning/best_config.json
```

## Result — single-window (5 seeds × 500 ep, test 2023-01 → 2025-01)

Full baseline panel, point estimates (deterministic, sorted by Sharpe):

| Baseline | Sharpe | Total return |
|---|---|---|
| SPY/QQQ 60/40 | +1.980 | +78.11% |
| SPY Buy&Hold | +1.917 | +63.36% |
| 60/40 SPY/AGG | +1.870 | +37.71% |
| Risk Parity | +1.694 | +41.46% |
| Equal Weight | +1.626 | +42.05% |
| Rolling MVO-LW (max-Sharpe) | +1.612 | +46.13% |
| Max Sharpe MVO (static) | +1.463 | +62.91% |
| Rolling MVO-LW (min-var) | +1.340 | +30.79% |
| Min Variance (static) | +1.049 | +30.51% |
| Momentum 12-1 | +0.890 | +30.38% |

**SAC agent** (5 seeds, net of cost): Sharpe **+0.082 ± 0.654** [95% CI −0.468,
+0.631]; gross (pre-cost) **+1.387 ± 0.270** — a **1.305 gross→net cost drag**
at mean turnover 0.286/step. Mean active share 0.335; 0% of seeds near-uniform
(the policy is doing something, just not enough to survive costs).

Per-seed JK–Memmel vs each baseline (ΔSharpe annual, n significant / 5 seeds):

| vs | ΔSharpe(annual) | n sig / 5 | median p |
|---|---|---|---|
| Equal Weight | −1.544 ± 0.654 | 5/5 | 5.7e-12 |
| **SPY/QQQ 60/40 (strongest, HEADLINE)** | **−1.838 ± 0.646** | **4/5** | **4.3e-04** |
| SPY Buy&Hold | −1.775 ± 0.646 | 4/5 | 7.1e-05 |
| 60/40 SPY/AGG | −1.728 ± 0.646 | 4/5 | 2.6e-04 |
| Risk Parity | −1.612 ± 0.654 | 5/5 | 3.8e-12 |
| Rolling MVO-LW (max-Sharpe) | −1.530 ± 0.654 | 4/5 | 2.8e-05 |
| Max Sharpe MVO (static) | −1.381 ± 0.654 | 3/5 | 1.8e-02 |
| Rolling MVO-LW (min-var) | −1.259 ± 0.654 | 4/5 | 3.2e-06 |
| Min Variance (static) | −0.967 ± 0.654 | 2/5 | 5.6e-02 |
| Momentum 12-1 | −0.808 ± 0.654 | 3/5 | 3.2e-02 |

Deflated Sharpe Ratio (n_trials=50): **0.0193** — up from Phase 1–3's ≈0.004–
0.0045, but still nowhere near surviving the multiple-testing haircut. Two
weakest baselines (Momentum 12-1, static Min Variance) are the only ones where
the per-seed JK isn't majority-significant, and even those have bootstrap CIs
that only barely fail to exclude zero (they don't help the agent's case, they
just aren't as harsh a bar).

## Result — walk-forward (3 seeds × 9 folds × 200 ep/fold, full 2019-04→2025-01 span)

Per-regime, agent NET Sharpe vs equal-weight and vs the strongest baseline that
regime:

| Regime | Agent NET Sharpe | EW Sharpe | k/N>EW | Best-baseline Sharpe | k/N>best | sig vs best |
|---|---|---|---|---|---|---|
| COVID crash/recovery | +0.997±0.255 | +2.387 | 0/6 | +2.880 | 0/6 | 3/6 |
| 2021 bull | −0.917±0.619 | +0.708 | 0/6 | +1.261 | 0/6 | 5/6 |
| 2022 bear (rate shock) | −1.428±1.570 | +0.063 | 0/6 | +0.185 | 0/6 | 3/6 |
| 2023-24 recovery/AI | −0.204±2.223 | +1.920 | 0/9 | +2.378 | 0/9 | 7/9 |

**Overall (27 folds):** NET Sharpe **−0.368 ± 1.742** [−1.017, +0.280] (gross
+1.260); beats EW in **0/27**, 25/27 JK-significant vs EW (median p=2.0e-4);
beats the strongest baseline in **0/27**, 18/27 JK-significant vs strongest
(median p=6.6e-3); ΔSharpe(annual) vs EW = −1.709±0.694, vs strongest baseline =
**−2.079±0.847** (a bigger deficit — the tougher bar sharpens, not softens, the
conclusion). DSR = 0.0010.

Baseline panel tested: Equal Weight, SPY Buy&Hold, 60/40 SPY/AGG, Risk Parity,
Rolling MVO-LW (min-var), Rolling MVO-LW (max-Sharpe).

## Interpretation

Phases 2–3's finding — the agent loses to equal-weight net of cost, everywhere
— **survives contact with a much harder benchmark set.** Against the single
strongest baseline per fold/seed, the deficit is *larger*, not smaller
(−2.08 vs −1.71 annualized Sharpe overall in walk-forward; −1.84 vs −1.54 in
the single-window). Gross Sharpe (+1.39 single-window, +1.26 walk-forward
pooled) is genuinely competitive with the baseline panel pre-cost — this is not
a signal problem. The cost drag at ~0.29–0.4/step turnover is what erases it.
**Phase 4 does not change the diagnosis, it strengthens the evidence for it:**
turnover, not signal, is the binding constraint, and Phase 5's cost-aware
reward / turnover penalty is the only lever that can plausibly move the net
result.

## Out of scope (per PHASE3_HANDOVER.md §8.7, unchanged)
Entropy/reward fixes and the turnover penalty (I-6, I-10 → Phase 5);
reproducibility nondeterminism (§12 → Phase 5); README/doc rewrite,
state-dim/FinRL fixes (I-9 → Phase 6); longer-proxy HPO re-tune (I-8 → Phase
5/6).
