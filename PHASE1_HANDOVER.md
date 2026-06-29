# SAC Portfolio Optimization — Phase 1 Handover & Phase 2 Brief

**Paste this whole document into a fresh Claude chat to continue the project.**
It contains everything needed: the project context, what Phase 0 and Phase 1
changed and why, the exact environment that works, the current honest CI-backed
results, and a complete, actionable specification for Phase 2 (leakage removal).

- **Repo:** https://github.com/dhruvi002/dynamic-portfolio-optimization
- **Local path (Mac):** `~/Documents/Projects/finrl/dynamic-portfolio-optimization`
- **Conda env:** `portfolio-rl` (Python 3.10, macOS arm64, CPU-only)
- **Phase 0 done:** June 2026 · **Phase 1 done:** June 2026
- **HEAD at handover:** `601f3ea` (Phase 1 complete)
- **Owner:** Dhruvi (MS-CS student targeting entry-level Data / ML / Applied-Research / GenAI roles)

---

## 0. Standing instructions for the assistant (read first)

1. **Honest reporting over big numbers.** Every headline metric ships with a
   confidence interval and a significance test. Never re-introduce a bare
   "+24%"/"+47%" headline. If an edge is not statistically significant, say so
   plainly. Report **net-of-cost** metrics (see §5.4 — this bit people once already).
2. **Reproducibility before results.** Nothing is re-measured until the run is
   seeded, version-pinned, and stamped (Phase 0). Keep it intact.
3. **Fix leakage before re-benchmarking — that is exactly Phase 2.** Do not
   produce "prettier wrong numbers"; remove the leak first, then re-run the
   Phase 1 harness unchanged.
4. **CPU-first.** Model is small (128–256-wide MLP, 30 assets). A 5-seed × 500-ep
   sweep is ~3.5–4 h wall-clock on a MacBook Air (each seed ~40–48 min). Only
   10+-seed sweeps or HPO benefit from Colab Pro.
5. **Git commits: do NOT add Claude / Sonnet / Opus as a co-author.** Commit
   regularly, authored as the user only. No `Co-authored-by` trailers.
6. **Work on the real repo**, not a scratch copy. Give exact, copy-pasteable
   commands and wait for output before moving on. The user runs training in
   their own terminal.
7. **Use the existing harness.** `experiments/multi_seed.py` (`make evaluate`)
   is the measurement instrument; Phase 2 changes the *pipeline it measures*, not
   the harness.

---

## 1. Project in one paragraph

A deep reinforcement-learning portfolio optimizer for the Dow-30. A **Soft
Actor-Critic (SAC)** agent with a **Dirichlet policy on the K-simplex** allocates
weights across 30 assets. The environment is a hand-rolled Gymnasium env (FinRL-
*compatible*, not FinRL-powered). The codebase is strong ML engineering — twin
critics, a Welford running normalizer that freezes at eval, chronological
train/val/test split, checkpoint selection on validation Sharpe, transaction +
slippage costs applied identically to agent and baselines, an optional cross-asset
transformer encoder, and an optional FinBERT sentiment channel. There is also a
walk-forward CV module and a Ray Tune HPO runner. **The original headline ("+24%
Sharpe / +47.3% return over equal-weight") rested on a single unseeded backtest.**
Phases 0–1 replaced it with a reproducible, multi-seed, CI-backed, significance-
tested evaluation. The honest verdict so far (still pre-leakage-fix): **after
costs the agent does not beat equal-weight.**

---

## 2. Issue register (full remediation backlog)

Severity: **H** = blocks the headline claim, **M** = materially weakens credibility, **L** = polish.

| ID | Issue | Sev | Phase | Status |
|----|-------|-----|-------|--------|
| I-1 | No global seeding (torch/numpy/random) → not reproducible | H | 0 | ✅ DONE |
| I-2 | Single backtest/seed/regime; no significance test or CI | H | 1, 3 | ✅ DONE (Phase 1 core); regimes → Phase 3 |
| I-3 | HPO leaks the test set (tunes on full 2019–2025 incl. test) | H | **2** | ⬜ TODO |
| I-4 | Survivorship / look-ahead universe (SHW back-filled to 2019) | H | **2** | ⬜ TODO |
| I-5 | In-sample vs out-of-sample gap unexplained; no turnover/concentration/active-share | H | 1, 5 | ✅ DONE (diagnosed + plotted) |
| I-6 | Entropy temperature collapses (alpha→~4e-4); max-entropy rationale inert | M | 1, 5 | ✅ MEASURED (fix → Phase 5) |
| I-7 | Weak/strawman baselines; static one-shot MVO; SPY buy-and-hold missing | M | 4 | ⬜ TODO |
| I-8 | Config ambiguity (tuned vs default); 15-ep HPO proxy may not transfer to 500-ep | M | 2, 6 | ◑ PARTIAL (Phase 0) |
| I-9 | Doc/code mismatches (tree, state-dim 180 vs 210, FinRL claim, missing modules) | L | 6 | ⬜ TODO |
| I-10 | Tiny reward signal (1e-4 scaling on log returns); learning-signal not validated | M | 5 | ⬜ TODO |
| I-11 | Headline overstates precision (relative '+24%' on a point estimate) | M | 1, 6 | ✅ DONE (CI-backed now) |

Dependencies: **0** gates everything → **1** built the measurement harness →
**2** removes leakage (must precede any headline re-benchmark) → **3, 4, 5** can
partially overlap → **6** closes out (docs + honest claims).

---

## 3. What Phase 0 did (COMPLETED — context)

Made every run deterministic and self-documenting. Added `utils/seeding.py`
(`set_global_seed`), `utils/run_meta.py` (`write_run_meta` → `run_meta.json` with
UTC time, git SHA + `-dirty`, seed, config, device, Python/platform, pinned lib
versions), `environment.yml`, `requirements.lock.txt`, and a `Makefile`. Wired
`--seed` through `main.py` / `utils/trainer.py` / `PortfolioEnv`. Removed broken
unused deps (`finrl==0.3.6`, `pyfolio-reloaded`), added `scipy`, upgraded
`yfinance` to 1.5.1 (for `multi_level_index`). Acceptance: two same-seed runs
produce identical logs (`make repro`); every artifact dir gets a `run_meta.json`.

---

## 4. What Phase 1 did (COMPLETED — this is the new work)

**Objective:** replace the single point estimate with a statistically defensible
evaluation — multiple seeds, confidence intervals, a significance test against
the benchmark, overfitting-aware Sharpe diagnostics, and policy-behavior
diagnostics. Resolved **I-2 (core)**, **I-5**, **I-11**; diagnosed **I-6**.

### 4.1 New files

**`utils/significance.py`** — significance & overfitting math (pure numpy/scipy):
- `jobson_korkie_memmel(returns_a, returns_b)` — Sharpe-difference test for two
  *correlated* return series with the **Memmel (2003)** variance correction
  (agent and equal-weight are both long-DJ30, highly correlated). Returns
  per-period & annualized Sharpes, the difference, correlation, z-stat, p-value.
- `sharpe_diff_bootstrap_ci(...)` — stationary (Politis–Romano) **block bootstrap**
  CI on the Sharpe difference; paired blocks preserve cross-correlation.
- `probabilistic_sharpe_ratio(...)`, `expected_max_sharpe(...)`,
  `deflated_sharpe_ratio(...)` — **Bailey & López de Prado** PSR/DSR: haircut the
  Sharpe for sample length, skew/kurtosis, and number of trials.
- `probability_of_backtest_overfitting(...)` — CSCV **PBO** (optional; needs a
  matrix of per-trial return series).
- **Convention:** test statistics use *per-period* (non-annualized) Sharpe;
  annualization cancels in the z-stat. `annualize_sharpe` is for display only.

**`utils/diagnostics.py`** — policy-behavior & aggregation:
- `returns_from_values(pv)` — **net-of-cost** daily returns from the value path.
- `reconcile_returns(returns_gross, pv)` — explains the total_return vs ann_return
  discrepancy (see §5.4).
- `turnover_series`, `hhi_series`, `active_share_series`, `policy_diagnostics(env)`
  — Σ|Δw|, HHI = Σw², active share = 0.5·Σ|w−1/N|, plus a `near_uniform` flag.
- `bootstrap_ci(...)`, `aggregate_metrics(per_seed, ...)`, `format_aggregate_table`
  — per-seed mean ± std + bootstrap 95% CI.

**`experiments/multi_seed.py`** — the harness (`make evaluate`). Trains +
backtests across N seeds; recomputes the agent's Sharpe family on the **net**
value path (and reports `gross_sharpe` beside it); runs per-seed JK–Memmel
(primary) + pooled + bootstrap + DSR vs equal-weight; computes IS & OOS policy
diagnostics; logs the α / policy-entropy trajectory; persists per-seed CSV/JSON,
`per_seed_returns.npz`, per-seed normalizers, a `run_meta.json`, and two figures.

**`test/test_significance.py`, `test/test_diagnostics.py`** — 22 analytic unit
tests for the new math (identical-series → p≈1; clear-difference → significant;
turnover/HHI/active-share on known weights; DSR haircut grows with trials; PBO
in [0,1] and lower when one strategy dominates; bootstrap CI brackets the point).

### 4.2 Modified files

- **`agent/sac.py`** — `update()` now also returns `policy_entropy` (−E[log π]),
  additive, no behavior change; lets the entropy collapse be logged/plotted.
- **`utils/plotting.py`** — `plot_diagnostics_panel` (one-page: turnover, HHI,
  active share, IS-vs-OOS equity curves) and `plot_alpha_entropy`.
- **`Makefile`** — `make evaluate` (`SEEDS="0 1 2 3 4" EPISODES=500`).
- **`test/test_all.py`** — fixed two pre-existing **incorrect assertions** (the
  code was right): max-drawdown expected `-20/110` not `-90/110`; the
  "positive returns" Sharpe test used constant (zero-variance) returns.

### 4.3 The total_return vs ann_return reconciliation (was handover §5)

Two effects made `ann_return` look positive while `total_return` was negative:
1. **Costs excluded from the agent's return array.** The env deducts
   transaction+slippage from `portfolio_value` directly, but `info["port_return"]`
   (which fed `backtest()`'s metrics) is **gross** of costs. Baselines were never
   affected — they derive returns from the value path.
2. **Arithmetic vs geometric annualization.** `(1+mean)^252−1` ignores
   volatility drag; the realized geometric path sits ~0.5·σ² lower.

`reconcile_returns()` prints all three side by side: `ann_return_arith_gross`
(legacy/misleading), `ann_return_arith_net` (costs included), `ann_return_geom_net`
(honest, consistent with `total_return`).

### 4.4 The same bug in the Sharpe family — and the fix

`backtest()` computes Sharpe/Sortino/Calmar from the **gross** return array, so
they were inflated whenever turnover was high. **`_net_metrics()` in the harness
recomputes them from the net value path and reports `gross_sharpe` alongside.**
The gross-minus-net Sharpe gap is now a direct read-out of the transaction-cost
drag — and it is the central finding (§5).

### 4.5 Significance methodology decision

**Per-seed JK–Memmel is the PRIMARY statement** — each seed is a genuine,
full-length out-of-sample track record. Headline = "k/N seeds significant" plus
mean ± std of ΔSharpe across seeds. The pooled JK (on the cross-seed-*averaged*
series) and the bootstrap CI are kept only as **optimistic cross-checks** —
averaging shrinks the standard error and overstates the z-stat, so they must not
be quoted as the headline p-value. (Keys in `significance.json`:
`per_seed_jk`, `per_seed_summary`, `jobson_korkie_memmel_pooled_optimistic`,
`bootstrap_sharpe_diff`, `deflated_sharpe_ratio`.)

### 4.6 Phase 1 commits (on `main`, authored Dhruvi, no co-author)

```
59f6566  Phase 1: honest evaluation harness (multi-seed CIs, significance, diagnostics)
600d707  Fix two incorrect assertions in test/test_all.py
7370355  Phase 1 fix: report net-of-cost Sharpe family (was using gross returns)
601f3ea  Phase 1: make per-seed JK the primary significance statement
```

### 4.7 Acceptance criteria — status

- ✅ Headline = mean ± std and 95% CI across **5 seeds** for every metric.
- ✅ p-value / CI for SAC-minus-equal-weight Sharpe, test named (Jobson–Korkie–
  Memmel, per-seed primary; bootstrap + pooled as cross-checks).
- ✅ One-page diagnostic figure (turnover, HHI, active share, IS-vs-OOS equity).
- ✅ DSR computed; PBO available.
- ✅ Alpha/entropy trajectory logged + plotted.
- ✅ total_return vs ann_return reconciled; net-vs-gross Sharpe exposed.
- ✅ All committed (no co-author); `run_meta.json` accompanies the batch.

---

## 5. Current honest result (Phase 1: 5 seeds, 500 episodes, tuned config)

Test window **2023-01-02 → 2025-01-30** (544 trading days), $1M start,
0.1% transaction cost + 0.1% slippage. Seeds `[0,1,2,3,4]`. Metrics are
**net of cost** unless labelled "gross". (Reproduce: `make evaluate`.)

### 5.1 Headline — SAC agent (mean ± std, bootstrap 95% CI across 5 seeds)

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| **Sharpe (net)** | **+0.264** | 0.446 | [−0.062, +0.692] |
| Sharpe (gross, pre-cost) | +1.593 | 0.124 | [+1.491, +1.694] |
| Sortino (net) | +0.396 | 0.672 | [−0.092, +1.041] |
| Calmar (net) | +0.330 | 0.574 | [−0.037, +0.904] |
| Max Drawdown | −15.5% | 3.5% | [−17.8%, −12.0%] |
| **Total Return** | **+5.7%** | 11.9% | [−2.9%, +17.2%] |
| Ann. Return (geom, net) | +2.4% | 5.2% | [−1.4%, +7.4%] |
| Ann. Volatility | 11.6% | 0.5% | [11.2%, 12.1%] |
| Win Rate | 49.9% | 1.5% | [48.7%, 51.2%] |
| Mean Turnover / step | 0.309 | 0.122 | [0.187, 0.395] |
| Mean HHI (1/N=0.033) | 0.072 | 0.024 | [0.050, 0.092] |
| Mean Active Share | 0.335 | 0.123 | [0.212, 0.420] |
| In-sample Sharpe (net) | +0.031 | 0.333 | [−0.234, +0.323] |

### 5.2 Per-seed (net), showing the cost/turnover story

| Seed | Net Sharpe | Total Return | Turnover | Active Share | Near-uniform |
|------|-----------|--------------|----------|--------------|--------------|
| 0 | −0.125 | −4.37% | 0.396 | 0.386 | no |
| 1 | **+1.096** | **+27.89%** | **0.077** | 0.099 | **yes** |
| 2 | +0.159 | +2.54% | 0.295 | 0.336 | no |
| 3 | −0.110 | −4.20% | 0.408 | 0.430 | no |
| 4 | +0.298 | +6.54% | 0.369 | 0.426 | no |

### 5.3 Baselines (deterministic; same costs)

| Strategy | Sharpe | Total Return |
|----------|--------|--------------|
| Equal Weight | 1.494 | +40.05% |
| SPY/QQQ 60/40 | 1.980 | +78.11% |
| Momentum 12-1 | 0.615 | +20.67% |
| Min Variance | 1.085 | +31.89% |
| Max Sharpe MVO | 1.596 | +72.85% |

### 5.4 Significance — SAC minus Equal-Weight Sharpe

- **PRIMARY (per-seed JK–Memmel):** **5/5 seeds significantly worse** at 5%
  (median p = 1.03e-9); ΔSharpe(annual) = **−1.230 ± 0.446** across seeds.
- **Deflated Sharpe Ratio** (n_trials=50): **DSR = 0.137** — low; the result does
  not survive a multiple-testing haircut, consistent with no genuine edge.
- Cross-checks (optimistic, do not quote as headline): pooled JK z = −10.4,
  p ≈ 0; bootstrap 95% CI on ΔSharpe(annual) = [−1.455, −1.005] (excludes 0).

### 5.5 Interpretation (the finding)

The agent finds signal **gross** (Sharpe ≈ 1.59, on par with Max-Sharpe MVO),
but ~0.31/step turnover imposes a **−1.33 Sharpe cost drag**, pulling **net**
performance to ≈ 0 and well below equal-weight (1.49). Seed 1 is the natural
experiment: it converged to a near-equal-weight policy (turnover 0.077) and was
the only seed with a respectable net Sharpe (1.10) — **trading less helped.**
In-sample net Sharpe (0.03) is barely positive, so this is an
underfitting/instability story (high seed variance, net Sharpe −0.13 → +1.10),
not classic overfitting. The α→~4e-4 entropy collapse is confirmed.

**Caveats:** (1) high seed variance is a real limitation — disclose it. (2) These
numbers are **pre-leakage-fix** (I-3, I-4 still present) — Phase 2 removes the
leak and re-runs; the apparent gross edge is expected to shrink further.

---

## 6. The environment that works (authoritative)

Conda env **`portfolio-rl`**, Python **3.10**, macOS arm64, CPU. Top-level pins:

```
torch==2.2.2        numpy==1.26.4      pandas==2.2.2       scipy==1.13.1
gymnasium==0.29.1   yfinance==1.5.1    ray[tune]==2.9.3    hyperopt==0.2.7
matplotlib==3.8.4   seaborn==0.13.2    tensorboard==2.16.2 tqdm==4.66.4
ta==0.11.0          transformers==4.40.2
```

`finrl` and `pyfolio-reloaded` are intentionally **excluded**. **`pytest` is a
test-only dep — `pip install pytest` then `make lock` if it is missing.** After a
clean install always run `make lock`. `make test` runs the full suite (93 tests).

---

## 7. Repo map (key files)

```
main.py                      # CLI: train / tune / backtest (+ --seed)
config.py                    # date windows: DOWNLOAD/TRAIN/TRAIN_PROPER/VAL/TEST
agent/sac.py                 # SACAgent, DirichletActor, Critic, ReplayBuffer,
                             #   AssetTransformerEncoder; update() now logs policy_entropy
environment/portfolio_env.py # Gymnasium env; reset(seed=); step() info has
                             #   port_return (GROSS of cost), turnover; history has
                             #   portfolio_value (NET), weights, returns (gross)
                             #   DJ30_TICKERS includes SHW (survivorship — I-4)
utils/trainer.py             # train(...seed=), backtest(...) — backtest metrics are
                             #   GROSS; use net via diagnostics/_net_metrics (Phase 1)
utils/metrics.py             # compute_sharpe/sortino/calmar/max_drawdown/all_metrics
utils/baselines.py           # equal_weight, spy_qqq, momentum_12_1, min_variance,
                             #   max_sharpe_mvo (value-path returns, net of cost)
utils/walk_forward.py        # expanding-window CV — EXISTS, not yet wired to CLI (Phase 3)
utils/normalizer.py          # RunningNormalizer (Welford; train()/eval(); skips weights)
utils/seeding.py             # Phase 0
utils/run_meta.py            # Phase 0
utils/significance.py        # NEW (Phase 1): JK–Memmel, bootstrap CI, PSR/DSR, PBO
utils/diagnostics.py         # NEW (Phase 1): turnover/HHI/active share, CI aggregation,
                             #   returns_from_values, reconcile_returns
utils/plotting.py            # + plot_diagnostics_panel, plot_alpha_entropy (Phase 1)
experiments/multi_seed.py    # NEW (Phase 1): the honest evaluation harness (make evaluate)
experiments/results/         # per_seed_results.csv, aggregate_metrics.json,
                             #   significance.json, baselines.json, run_meta.json,
                             #   per_seed_returns.npz, diagnostics_panel.png,
                             #   alpha_entropy_trajectory.png, checkpoints/seed_*.pt + *_normalizer.pkl
tuning/tune_runner.py        # Ray Tune HPO — *** THE HPO TEST-SET LEAK lives here (I-3, Phase 2) ***
tuning/best_config.json      # tuned HPs (LEAKY — regenerate in Phase 2)
data/pipeline.py             # download_data, add_technical_indicators, split_data,
                             #   three_way_split; DJ30_TICKERS (survivorship — I-4)
data/sentiment_pipeline.py   # FinBERT sentiment (optional; state 180→210)
test/                        # pytest suite incl. test_significance.py, test_diagnostics.py
Makefile, environment.yml, requirements.lock.txt
PHASE1_NOTES.md              # Phase 1 design notes (companion to this doc)
```

Useful signatures:
- `from utils.diagnostics import returns_from_values, policy_diagnostics, aggregate_metrics`
- `from utils.significance import jobson_korkie_memmel, sharpe_diff_bootstrap_ci, deflated_sharpe_ratio`
- `backtest(agent, env, normalizer=None) -> dict` (GROSS metrics — wrap with net!)
- `three_way_split(df, train_start, train_end, val_start, val_end, test_start, test_end)`

---

## 8. PHASE 2 — Remove leakage (DO THIS NEXT)

**Objective:** make the pipeline leakage-free so the Phase 1 numbers can be
trusted, then re-run the Phase 1 harness unchanged. Resolves **I-3** and **I-4**
(and part of **I-8**). **Budget:** ~2–4 days part-time (one HPO sweep + one
5-seed eval, both overnight-able on the Mac; HPO benefits from Colab Pro).

### 8.1 Task A — kill the HPO test-set leak (I-3) in `tuning/tune_runner.py`

**The leak (confirmed):** `_train_trial()` loads the *entire*
`data/processed_data.parquet` (2019-04 → 2025-01, **including the 2023–2025 test
window**), builds **one** `PortfolioEnv` over the whole series, trains on it, and
reports `compute_sharpe` on that **same full series**. ASHA/HyperOpt then select
`best_config.json` using test-window performance → the test set influenced the
hyperparameters that §5 reports on. (It also uses gross returns and is unseeded.)

**Fix:**
1. In `_train_trial`, after loading the parquet, call `three_way_split(...)` with
   the windows from `config.py` and build the **training env on `train_df` only**
   (`TRAIN_START`→`TRAIN_PROPER_END`) and a **val env on `val_df`**
   (`VAL_START`→`TRAIN_END`). **Never construct anything over `test_df`.**
2. Train the short HPO trial on `train_df`; compute the **reported objective as
   the deterministic NET-of-cost validation Sharpe** (run `utils.trainer.backtest`
   on the val env with a `RunningNormalizer`, then take the net Sharpe via
   `utils.diagnostics.returns_from_values` / `compute_sharpe`). Match the Phase 1
   "net Sharpe" definition so HPO optimizes the metric we actually report.
3. **Seed each trial** (`set_global_seed(config["seed"])`, thread a seed through
   the search space or fix one) for reproducible HPO.
4. Add a hard **guard**: assert the env's max date ≤ `TRAIN_END` inside the trial,
   raising if any test-window row is present.
5. Re-run HPO (`python main.py --mode tune` or `python -m tuning.tune_runner`,
   `num_samples>=50`) to produce a **new, leak-free `tuning/best_config.json`**.
   Archive the old one as `tuning/best_config_LEAKED.json` for the record.

**Unit-test guard (new `test/test_no_leak.py`):** load processed data, monkeypatch
or inspect the env built by the tuning path, and assert it contains **no dates in
`[TEST_START, TEST_END]`**. This is a regression guard the brief requires.

### 8.2 Task B — fix the survivorship / look-ahead universe (I-4)

**The problem:** `DJ30_TICKERS` (in both `data/pipeline.py` and
`environment/portfolio_env.py`) is the **current** index membership with **SHW
back-filled to 2018** (SHW only joined the DJIA in 2024, replacing WBA). Using
2024-knowledge constituents for the 2019–2022 training period is look-ahead /
survivorship bias; SHW/DOW/CRM etc. that changed membership mid-sample
contaminate the universe.

**Fix — pick one and document it plainly:**
- **(Recommended, Phase-2-sized) Disclosed fixed neutral universe.** Choose the
  subset of tickers that were **continuously index members across the entire
  2018–2025 window**, drop the rest (SHW and any other mid-sample joiner/leaver),
  and state in the README/results: *"fixed liquid large-cap universe of N names,
  not the live DJ-30; chosen to avoid survivorship/look-ahead bias; excluded: …
  because …"*. Re-process data for that universe.
- **(Stretch) Point-in-time membership.** Build a `date → constituents` table from
  historical DJIA changes and have the env restrict the active universe per date.
  Rigorous but heavier (needs the full constituent-change history).

**Implementation for the recommended option:** add `UNIVERSE` to `config.py`,
replace the hard-coded `DJ30_TICKERS` references in `data/pipeline.py` and
`PortfolioEnv` with it, delete `data/processed_data.parquet` and
`data/raw_data.parquet`, and rebuild (running any `--mode train`/`tune` once
re-downloads + re-processes). Keep the SPY/QQQ baseline as-is.

### 8.3 Re-measure under the corrected pipeline

1. Rebuild data for the leak-free universe.
2. Re-run HPO → new `best_config.json` (validation-only objective).
3. `make evaluate` (the **unchanged** Phase 1 harness) with the new config.
4. Compare leak-free vs Phase 1 (leaky) §5 numbers with the same CI/significance
   framing. **Expected:** gross Sharpe drops (HPO no longer peeks at test) and the
   net-loses-to-EW finding persists or strengthens. Report honestly either way.

### 8.4 Acceptance criteria (Phase 2)

- `tuning/` provably never reads `[TEST_START, TEST_END]` rows — unit-test-guarded
  (`test/test_no_leak.py`) and documented.
- HPO objective computed on **validation only**, net of cost, seeded; new
  `best_config.json` regenerated; old archived as `*_LEAKED.json`.
- Universe is leak-free (disclosed fixed set or point-in-time), documented with
  the excluded names and the reason.
- Phase 1 harness **re-run** on the corrected pipeline; fresh CI-backed §5 table +
  significance + `run_meta.json` committed.
- All commits authored as Dhruvi, **no co-author**.

### 8.5 Suggested commands

```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization

# after editing tune_runner.py + universe and deleting the stale parquet:
python main.py --mode train --episodes 1 --seed 42      # rebuilds data once (or write a tiny build script)
python -m tuning.tune_runner                            # leak-free HPO → new best_config.json
make test                                               # incl. the new no-leak guard
make evaluate                                           # 5 seeds × 500 ep on the corrected pipeline
```

### 8.6 Explicitly OUT of scope for Phase 2

- Entropy mechanism / reward-scaling changes (I-6, I-10) → **Phase 5**.
- New baselines: SPY buy-and-hold, 60/40 SPY/AGG, risk parity, Ledoit-Wolf MVO
  (I-7) → **Phase 4**.
- Wiring `utils/walk_forward.py` into the CLI / per-regime tables → **Phase 3**.
- README/doc rewrite, state-dim & FinRL-claim fixes (I-9) → **Phase 6**.

---

## 9. Roadmap beyond Phase 2 (context, not action)

- **Phase 3** — wire `utils/walk_forward.py` into the CLI; expanding-window folds
  across regimes (2020 COVID, 2022 bear, 2023–24 recovery); per-regime table;
  multi-seed per fold (reuse the Phase 1 aggregation + significance).
- **Phase 4** — baseline overhaul: SPY buy-and-hold, 60/40 SPY/AGG, risk parity,
  rolling re-estimated MVO with Ledoit–Wolf shrinkage; same costs, seeds, CIs.
- **Phase 5** — entropy/learning-signal: entropy floor / scheduled α / re-derived
  target entropy; reward-scaling sweep; **explicitly target the turnover problem
  from §5** (e.g., turnover penalty / cost-aware reward) since costs are the
  binding constraint; ablate MLP vs transformer and with/without FinBERT.
- **Phase 6** — sync docs to code (state-dim 180 vs 210, "FinRL-compatible" not
  -powered, repo tree), rewrite results around the CI-backed numbers, add a
  Limitations section and a one-command reproduction recipe.

Then a **Capstone Roadmap** (only after results are trustworthy): Docker, W&B +
Hydra tracking, GitHub Actions CI, a deployed Streamlit/Gradio demo, algorithm
breadth (PPO/TD3 + CVaR risk-aware reward), and a GenAI extension (LLM
news/filings → signals, RAG over 10-Ks, LLM rebalance rationales).

---

## 10. Definition of Done (whole remediation effort)

- ✅ All runs seeded, version-pinned, commit+config stamped (Phase 0).
- ✅ Every headline metric = mean ± 95% CI across ≥5 seeds, with a named
  significance test vs equal-weight (Phase 1).
- ⬜ No leakage in HPO or universe; both unit-test-guarded and documented (Phase 2).
- ⬜ Results across ≥3 market regimes (Phase 3).
- ⬜ Strong baseline set (Phase 4).
- ◑ Policy behavior (turnover/concentration/active share) and IS/OOS gap explained
  (✅ done Phase 1); entropy mechanism functioning or no longer claimed (Phase 5).
- ⬜ README matches code, has a Limitations section + one-command repro (Phase 6).

---

## 11. Gotchas / lessons carried forward

- **Always report NET (value-path) Sharpe.** `utils/trainer.backtest()` returns
  **gross** metrics; the Phase 1 harness wraps it with `_net_metrics()`. Any new
  evaluation code must do the same or it will overstate performance (this exact
  bug produced an inconsistent "Sharpe +1.32 next to total_return −7%" table once).
- **Per-seed JK is primary**; pooled/bootstrap are optimistic cross-checks.
- A stray `diff --git a/checkpoints/best_agent.pt …` line sometimes prints during
  a run — benign terminal/git noise, **not** emitted by the harness; ignore it.
- `pytest` is not in the runtime lock; install it before `make test`.
- If git refuses with a stale `.git/index.lock`, remove that file and retry.
- Run artifacts (`checkpoints/`, `plots/`, `runs/`, `experiments/results/`) are
  regenerated each run; don't commit the large binaries — commit source + the
  small JSON/CSV/PNG summaries if you want them tracked.
