# Phase 2 Notes — Leakage Removal (I-3, I-4)

Companion to `PHASE1_HANDOVER.md` §8. Phase 2 makes the pipeline **leakage-free**
so the Phase 1 numbers can be trusted, then re-runs the **unchanged** Phase 1
harness (`make evaluate`). It resolves **I-3** (HPO test-set leak) and **I-4**
(survivorship / look-ahead universe), and part of **I-8** (config provenance).

Standing rules from the handover still hold: honest reporting over big numbers,
net-of-cost metrics, per-seed JK as the primary significance statement, CPU-first,
and commits authored as Dhruvi with **no co-author**.

---

## 1. I-3 — Kill the HPO test-set leak (`tuning/tune_runner.py`)

### The leak (before)
`_train_trial()` loaded the *entire* `data/processed_data.parquet`
(2019-04 → 2025-01, **including the 2023–2025 test window**), built **one**
`PortfolioEnv` over the whole series, trained on it, and reported `compute_sharpe`
on that **same full series**. ASHA/HyperOpt then selected `best_config.json` using
performance that **included the test window** → the test set influenced the
hyperparameters that the Phase 1 §5 table reports on. It was also computed on
**gross** returns and was **unseeded**.

### The fix (after)
- **Split first, never touch test.** New `_build_trial_envs()` calls
  `three_way_split()` with the canonical `config.py` windows and builds:
  - a **train** env on `[TRAIN_START, TRAIN_PROPER_END]` (2019-04 → 2021-12), and
  - a **val** env on `[VAL_START, TRAIN_END]` (2022-01 → 2022-12).

  Nothing is ever constructed over `test_df`.
- **Validation objective, net of cost.** `_val_net_sharpe()` runs a deterministic
  `backtest` on the val env with a frozen `RunningNormalizer`, then computes the
  Sharpe of the **value-path (net-of-cost) returns** via
  `diagnostics.returns_from_values` + `metrics.compute_sharpe`. This matches the
  Phase 1 harness's `_net_metrics` definition, so HPO optimizes the *exact* metric
  we report — not the inflated gross Sharpe.
- **Seeded trials.** Each trial calls `set_global_seed(config["seed"])` (fixed at
  42 in the search space) for reproducible HPO.
- **Hard leak guard.** Inside `_build_trial_envs`, both envs are asserted to have
  `max(date) ≤ TRAIN_END` and to contain **no** date in `[TEST_START, TEST_END]`;
  otherwise it raises `AssertionError("HPO LEAK GUARD: …")`.
- **Ray metric renamed** `sharpe → val_sharpe` (`TUNE_METRIC`) end-to-end
  (scheduler, search, best-result selection) so the optimized quantity is
  unambiguous.
- **Lazy Ray import.** `ray`/`ray.tune` are imported inside `run_tune()` and
  `_train_trial()` so the module and its trial helpers import without a full Ray
  install (lets the unit-test guard run in CI).

### Provenance
The old, leaky hyperparameters are archived verbatim at
`tuning/best_config_LEAKED.json`. Re-running HPO overwrites
`tuning/best_config.json` with the leak-free set.

### HPO engine — Ray Tune with a Ray-free fallback
Ray 2.9.3 is incompatible with the current `setuptools`/`pyarrow` (it imports the
removed `pkg_resources._vendor.packaging` and `pyarrow.PyExtensionType`). Rather
than pin a fragile dependency chain to tune a tiny CPU model, `run_tune()` tries
to import Ray and, if that fails for any reason, automatically falls back to
`run_tune_simple()` — a **sequential random search over the identical search
space** that scores each trial with the **same deterministic net-of-cost
validation Sharpe** (`_run_one_trial`, shared by both paths). The leakage
guarantees (train-only envs, hard test-window guard, seeded trials) are identical;
only the search strategy differs (random vs ASHA/TPE). This keeps Phase 2
unblocked and fully reproducible. If you later restore a Ray-compatible env
(`pip install "setuptools<71"` plus a compatible `pyarrow`), the Ray path is used
automatically with no code change.

---

## 2. I-4 — Survivorship / look-ahead universe

### The problem
`DJ30_TICKERS` was the **current** index membership with **SHW back-filled to
2018**. SHW joined the DJIA only in **Nov 2024**, so using it for the 2019–2022
training window is look-ahead bias; symmetrically, silently excluding names that
were dropped from the index (because they underperformed) is survivorship bias.

### The fix — disclosed fixed neutral universe
`config.UNIVERSE` is now the **24 names that were continuous DJIA members across
the entire 2018-01 → 2025-01 window**. It is **not** the live DJ-30 and carries no
foreknowledge of index changes. `config.EXCLUDED_FROM_DJ30` records the dropped
names and why:

| Excluded | Reason |
|---|---|
| AMGN, CRM, HON | joined DJIA 2020-08-31 (not members during 2019–mid-2020 training) |
| DOW | joined 2019-04-02, removed 2024-11-08 (not continuous across the window) |
| INTC | removed from DJIA 2024-11-08 (not a member through the window end) |
| SHW | joined DJIA 2024-11-08 (back-filling to 2018 is look-ahead bias) |

**Retained (24):** AAPL, AXP, BA, CAT, CSCO, CVX, DIS, GS, HD, IBM, JNJ, JPM, KO,
MCD, MMM, MRK, MSFT, NKE, PG, TRV, UNH, V, VZ, WMT.

`config.UNIVERSE` is the single source of truth; `PortfolioEnv.UNIVERSE`,
`PortfolioEnv.DJ30_TICKERS` (kept as a backward-compatible alias), and the
`DJ30_TICKERS` in `data/pipeline.py` and `data/sentiment_pipeline.py` all resolve
to it. The SPY/QQQ baseline is unchanged (separate download).

> **Note on the alternative.** Point-in-time membership (a `date → constituents`
> table that restricts the active universe per date) is more rigorous but heavier;
> the handover scopes it as a *stretch* goal beyond Phase 2. The disclosed fixed
> universe is the Phase-2-sized, defensible choice.

---

## 3. New / modified files

**Modified**
- `config.py` — added `UNIVERSE` (24) and `EXCLUDED_FROM_DJ30` (6, with reasons).
- `environment/portfolio_env.py` — `UNIVERSE` class attr from config; `DJ30_TICKERS`
  is now an alias; default `self.tickers` uses `UNIVERSE`.
- `data/pipeline.py`, `data/sentiment_pipeline.py` — `DJ30_TICKERS` ← `config.UNIVERSE`.
- `tuning/tune_runner.py` — leak-free `_build_trial_envs`, `_val_net_sharpe`,
  rewritten `_train_trial`; seeded trials; lazy Ray import; `TUNE_METRIC`;
  `_synthetic_df` now spans train+val+test (24 names).
- `README.md` — Phase 2 disclosure section; universe table; legacy "+24%" flagged
  as superseded.

**New**
- `test/test_no_leak.py` — 4 guards: universe excludes the 6 names; env/pipeline
  share the universe; tuning envs hold no test-window rows; the leak guard fires
  when test rows are injected.
- `tuning/best_config_LEAKED.json` — archived pre-fix hyperparameters.

---

## 4. Re-measurement procedure (run on the Mac, `portfolio-rl` env)

```bash
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization

# 0. Drop stale data so the 24-name universe is rebuilt from scratch
rm -f data/processed_data.parquet data/raw_data.parquet

# 1. Rebuild data for the leak-free universe (one short train re-downloads+processes)
python main.py --mode train --episodes 1 --seed 42

# 2. Regression tests, incl. the new no-leak guard (pytest is a test-only dep)
pip install pytest >/dev/null 2>&1 ; make test

# 3. Leak-free HPO -> new tuning/best_config.json (objective = net val Sharpe).
#    Uses Ray Tune if importable, else auto-falls-back to Ray-free random search.
python -m tuning.tune_runner            # or: python main.py --mode tune --tune-samples 50

# 4. Re-run the UNCHANGED Phase 1 harness on the corrected pipeline (~3.5-4 h)
make evaluate                            # SEEDS="0 1 2 3 4" EPISODES=500

# 5. Commit ONLY source + small summaries (authored Dhruvi, NO co-author)
git add tuning/best_config.json experiments/results/*.json experiments/results/*.csv
git commit -m "Phase 2: leak-free HPO config + re-measured results"
```

**What to expect / report honestly:** with HPO no longer peeking at test, the
gross edge should shrink; the Phase 1 finding (net loses to equal-weight after the
~0.31/step turnover cost drag) is expected to persist or strengthen. Compare the
new §5 table to the leaky one with the same CI / per-seed-JK / DSR framing, and
disclose the high seed variance either way.

---

## 5. Leak-free results (the Phase 2 finding)

Test window 2023-01-02 → 2025-01-30 (544 days), $1M start, 0.1% transaction +
0.1% slippage. 5 seeds [0–4] × 500 episodes, 24-name universe, leak-free HPO
config. Net of cost unless noted. (`make evaluate`.)

| Metric | Mean | Std | 95% CI |
|---|---|---|---|
| **Sharpe (net)** | **−0.081** | 0.743 | [−0.515, +0.657] |
| Sharpe (gross, pre-cost) | +1.489 | 0.155 | [+1.354, +1.625] |
| Total Return | −2.3% | 18.8% | [−13.2%, +16.5%] |
| Ann. Return (geom, net) | −1.5% | 8.3% | [−6.4%, +6.8%] |
| Max Drawdown | −19.8% | 5.6% | [−23.8%, −14.5%] |
| Mean Turnover / step | 0.352 | 0.152 | [0.200, 0.437] |
| In-sample Sharpe (net) | −0.331 | 0.496 | — |

Baselines (same costs): Equal-Weight **1.626** (+42.0%), SPY/QQQ 60/40 1.980,
Max-Sharpe MVO 1.463, Min-Variance 1.049, Momentum 12-1 0.890.

**Significance vs Equal-Weight** (PRIMARY = per-seed Jobson–Korkie–Memmel):
**4/5 seeds significantly worse** at 5% (median p = 3.7e-10); ΔSharpe(annual) =
**−1.706 ± 0.743**. **DSR = 0.004** — effectively zero; the result does not
survive the multiple-testing haircut. Cost drag: gross +1.489 → net −0.081, a
**−1.57 Sharpe** transaction-cost penalty at 0.35/step turnover.

### Phase 1 (leaky) vs Phase 2 (leak-free)

| | Phase 1 (leaky) | Phase 2 (leak-free) |
|---|---|---|
| Universe | 30 names, SHW back-filled | 24 continuous members |
| HPO objective | full series incl. test | validation-only, net of cost |
| **Net Sharpe** | +0.264 | **−0.081** |
| Gross Sharpe | +1.593 | +1.489 |
| Seeds worse than EW | 5/5 | 4/5 |
| **DSR** | 0.137 | **0.004** |

Removing both leaks pushed net Sharpe from marginally positive to negative and the
DSR from "low" to ≈0. The honest verdict is now **stronger**: after costs the agent
does not beat — it loses to — equal-weight, driven by the ~1.57 Sharpe cost drag,
and the apparent edge does not survive a multiple-testing correction. Gross Sharpe
also fell (1.59 → 1.49), exactly as expected once HPO can no longer see the test set.

**Caveats to disclose.** (1) Seed variance is high (net Sharpe std 0.74) — this is
why 4/5 (not 5/5) seeds reach significance; report it plainly. (2) The leak-free
HPO config drove the entropy temperature α *up* (to ≈6 by ep 500) rather than
collapsing it, raising turnover and widening the validation→test gap — a sign the
15-episode HPO proxy transfers poorly to 500 episodes (I-8) and that the entropy
mechanism needs attention (I-6). Both are explicitly **out of Phase 2 scope**
(Phase 5/6); they do not affect the leak-removal deliverable.

---

## 6. Phase 2 acceptance — status

- ✅ `tuning/` provably never reads `[TEST_START, TEST_END]` rows — guarded by
  `test/test_no_leak.py` and documented here.
- ✅ HPO objective = **validation-only, net-of-cost, seeded**; pipeline rewritten;
  old config archived as `best_config_LEAKED.json`; new leak-free
  `best_config.json` regenerated (50-trial sweep).
- ✅ Universe is leak-free (disclosed fixed 24-name set), documented with the
  excluded names and reasons (`config.EXCLUDED_FROM_DJ30`, README, this doc).
- ✅ Phase 1 harness re-run on the corrected pipeline; fresh CI-backed §5 table +
  significance + `run_meta.json` produced (see §5).
- ⬜ **Owner action:** commit `best_config.json` + `experiments/results/` summaries.

**Verdict:** Phase 2 complete. Both leaks removed and guarded; the leak-free
re-measurement shows the agent loses to equal-weight after costs (net Sharpe
−0.08, DSR 0.004) — an honest result, stronger than the pre-fix numbers.

---

## 7. Next — Phase 3 (context)

Wire `utils/walk_forward.py` into the CLI: expanding-window folds across regimes
(2020 COVID, 2022 bear, 2023–24 recovery), a per-regime table, multi-seed per fold
reusing the Phase 1 aggregation + significance. Do this **after** the Phase 2
re-measurement is committed, so regimes are evaluated on the leak-free pipeline.
