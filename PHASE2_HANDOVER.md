# SAC Portfolio Optimization — Phase 2 Handover & Phase 3 Brief

**Paste this whole document into a fresh Claude chat to continue the project.**
It contains everything needed: project context, what Phases 0–2 changed and why,
the exact environment that works (including the dependency landmines we hit), the
current honest leak-free results, and a complete, actionable specification for
Phase 3 (multi-regime walk-forward evaluation).

- **Repo:** https://github.com/dhruvi002/dynamic-portfolio-optimization
- **Local path (Mac):** `~/Documents/Projects/finrl/dynamic-portfolio-optimization`
- **Conda env:** `portfolio-rl` (Python 3.10, macOS arm64, CPU-only)
- **Phase 0:** June 2026 · **Phase 1:** June 2026 · **Phase 2:** June 2026
- **HEAD at handover:** `0f2daea` (Phase 2 complete)
- **Owner:** Dhruvi (MS-CS student targeting entry-level Data / ML / Applied-Research / GenAI roles)

---

## 0. Standing instructions for the assistant (read first)

1. **Honest reporting over big numbers.** Every headline metric ships with a
   confidence interval and a significance test. Never re-introduce a bare
   "+24%"/"+47%" headline. If an edge is not statistically significant, say so.
   Report **net-of-cost** metrics — `utils/trainer.backtest()` returns **gross**
   metrics; wrap them with the net value-path computation (this has bitten the
   project twice; see §12).
2. **Reproducibility before results.** Runs are seeded, version-pinned, and
   stamped (`run_meta.json`). Keep it intact. NOTE: there is a known residual
   nondeterminism on this Mac (§12) worth fixing early in Phase 3.
3. **No leakage — Phase 2 removed two leaks; do not re-introduce them.** The HPO
   trains on train-only and scores on validation-only (net of cost); the universe
   is a fixed continuous-membership set. Both are unit-test-guarded
   (`test/test_no_leak.py`). Any new evaluation must preserve this.
4. **CPU-first.** Model is small (MLP, 24 assets, state-dim 144). A 5-seed × 500-ep
   sweep is ~6–8 h wall-clock on a MacBook Air (each seed ~45–90 min depending on
   config/load). Walk-forward (Phase 3) is multiple folds × seeds — budget
   accordingly or reduce episodes per fold.
5. **Git: do NOT add Claude / Sonnet / Opus as a co-author.** Commit regularly,
   authored as the user only. No `Co-authored-by` trailers. Do **not** `git add -A`
   blindly — it sweeps in large `.pt`/`.npz`/`.pkl` binaries; the `.gitignore` now
   covers `checkpoints/ runs/ plots/ experiments/results/checkpoints/ *.pt *.pkl
   *.npz`. Commit source + the small JSON/CSV/PNG summaries explicitly.
6. **Work on the real repo**, not a scratch copy. Give exact, copy-pasteable
   commands and wait for output. The user runs training in their own terminal.
   **Their shell is zsh, which does NOT treat `#` as a comment by default** — put
   no inline `# comments` on command lines (they get passed as arguments and break
   the command). Keep commands comment-free or put comments on separate lines only
   if `setopt interactive_comments` is set.
7. **Use the existing harness.** `experiments/multi_seed.py` (`make evaluate`) is
   the single-window measurement instrument; Phase 3 builds the *walk-forward*
   analogue and should reuse its aggregation/significance helpers, not reinvent
   them.

---

## 1. Project in one paragraph

A deep reinforcement-learning portfolio optimizer for a fixed large-cap US equity
universe. A **Soft Actor-Critic (SAC)** agent with a **Dirichlet policy on the
K-simplex** allocates weights across the universe. The environment is a hand-rolled
Gymnasium env (FinRL-*compatible*, not FinRL-powered). The codebase is strong ML
engineering — twin critics, a Welford running normalizer that freezes at eval,
chronological train/val/test split, checkpoint selection on validation Sharpe,
transaction + slippage costs applied identically to agent and baselines, an
optional cross-asset transformer encoder, and an optional FinBERT sentiment channel.
There is also a walk-forward CV module and (now Ray-free) HPO runner. **The original
headline ("+24% Sharpe / +47.3% return over equal-weight") rested on a single
unseeded, leaky backtest.** Phases 0–2 replaced it with a reproducible, multi-seed,
CI-backed, significance-tested, **leakage-free** evaluation. **The honest verdict:
after costs the agent does not beat — it loses to — equal-weight (net Sharpe −0.08
vs 1.63; DSR 0.004).**

---

## 2. Issue register (full remediation backlog)

Severity: **H** = blocks the headline claim, **M** = materially weakens credibility, **L** = polish.

| ID | Issue | Sev | Phase | Status |
|----|-------|-----|-------|--------|
| I-1 | No global seeding → not reproducible | H | 0 | ✅ DONE |
| I-2 | Single backtest/seed/regime; no significance/CI | H | 1, 3 | ✅ DONE (Phase 1 core); **regimes → Phase 3** |
| I-3 | HPO leaks the test set | H | **2** | ✅ **DONE (Phase 2)** |
| I-4 | Survivorship / look-ahead universe (SHW back-filled) | H | **2** | ✅ **DONE (Phase 2)** |
| I-5 | IS-vs-OOS gap unexplained; no turnover/concentration/active-share | H | 1 | ✅ DONE |
| I-6 | Entropy temperature pathology (collapse → now *blow-up*; see §6) | M | 5 | ◑ MEASURED (fix → Phase 5) |
| I-7 | Weak/strawman baselines; SPY buy-and-hold missing | M | 4 | ⬜ TODO |
| I-8 | Config ambiguity; 15-ep HPO proxy may not transfer to 500-ep | M | 2, 6 | ◑ PARTIAL (leak-free now; transfer gap remains) |
| I-9 | Doc/code mismatches (state-dim **now 144**, FinRL claim, tree) | L | 6 | ⬜ TODO (partly addressed in README/notes) |
| I-10 | Tiny reward signal (1e-4 scaling); learning-signal not validated | M | 5 | ⬜ TODO |
| I-11 | Headline overstates precision | M | 1 | ✅ DONE |

Dependencies: **0** → **1** (measurement harness) → **2** (leak removal — done) →
**3, 4, 5** can partially overlap → **6** closes out (docs + honest claims).

---

## 3. What Phase 0 did (context)

Made every run deterministic and self-documenting. Added `utils/seeding.py`
(`set_global_seed`), `utils/run_meta.py` (`write_run_meta` → `run_meta.json` with
UTC time, git SHA + `-dirty`, seed, config, device, versions), `environment.yml`,
`requirements.lock.txt`, a `Makefile`. Wired `--seed` through the stack. Removed
broken deps (`finrl`, `pyfolio-reloaded`), added `scipy`, upgraded `yfinance`.

## 4. What Phase 1 did (context)

Replaced the single point estimate with a statistically defensible evaluation:
multiple seeds, bootstrap CIs, a significance test vs equal-weight, overfitting-aware
Sharpe diagnostics, and policy-behavior diagnostics. New files:
`utils/significance.py` (Jobson–Korkie–Memmel Sharpe-difference test, stationary
block-bootstrap CI, PSR/DSR, PBO), `utils/diagnostics.py` (net-of-cost
`returns_from_values`, turnover/HHI/active-share, cross-seed aggregation +
bootstrap CI), `experiments/multi_seed.py` (the `make evaluate` harness), and 22
analytic unit tests. **Convention:** test statistics use *per-period* Sharpe;
annualization is display-only. **Primary significance statement = per-seed JK–Memmel**
(each seed is a full-length OOS track record); pooled/bootstrap are optimistic
cross-checks only.

---

## 5. What Phase 2 did (THIS IS THE NEW WORK)

**Objective:** make the pipeline **leakage-free** so the Phase 1 numbers can be
trusted, then re-run the **unchanged** Phase 1 harness. Resolved **I-3**, **I-4**,
and the leak-free part of **I-8**. Full design notes: `PHASE2_NOTES.md`.

### 5.1 I-3 — killed the HPO test-set leak (`tuning/tune_runner.py`)

**The leak (before):** `_train_trial()` loaded the *entire*
`data/processed_data.parquet` (2019 → 2025, **including the 2023–2025 test window**),
built **one** env over the whole series, trained, and reported `compute_sharpe` on
that **same full series** (also gross, also unseeded). ASHA/HyperOpt then picked
`best_config.json` using test-window performance → the test set shaped the
hyperparameters Phase 1 §5 reported on.

**The fix:**
- New `_build_trial_envs()` calls `three_way_split()` with the `config.py` windows
  and builds a **train** env on `[TRAIN_START, TRAIN_PROPER_END]` and a **val** env
  on `[VAL_START, TRAIN_END]`. Never constructs anything over `test_df`.
- New `_val_net_sharpe()` computes the objective as the **deterministic net-of-cost
  validation Sharpe** (backtest on val env with a frozen `RunningNormalizer`, then
  Sharpe of the value-path returns) — matching the Phase 1 net-Sharpe definition.
- Each trial is **seeded** (`set_global_seed(config["seed"])`, fixed 42).
- A **hard leak guard** asserts both envs have `max(date) ≤ TRAIN_END` and contain
  no `[TEST_START, TEST_END]` rows, raising `AssertionError("HPO LEAK GUARD: …")`.
- Ray metric renamed `sharpe → val_sharpe` (`TUNE_METRIC`) end-to-end.
- Old leaky config archived as `tuning/best_config_LEAKED.json`; new leak-free
  `tuning/best_config.json` regenerated by a 50-trial sweep.

**Ray-free fallback (important).** Ray 2.9.3 is **broken** against current
`setuptools`/`pyarrow` in this env (it imports the removed
`pkg_resources._vendor.packaging` and `pyarrow.PyExtensionType`). Rather than pin a
fragile chain to tune a tiny CPU model, `run_tune()` now **try/excepts the Ray
import and auto-falls-back to `run_tune_simple()`** — a sequential random search
over the **same** search space with the **identical** net-validation objective and
the same leakage guards (`_run_one_trial` is shared by both paths). Ray and
HyperOpt imports are lazy, so the module imports (and unit-tests) without them. If a
Ray-compatible env is ever restored, the Ray path is used automatically.

**Unit guard:** `test/test_no_leak.py` (4 tests) — universe excludes the 6
mid-sample names; env/pipeline share the universe; the tuning envs hold no
test-window rows; the leak guard fires when test rows are injected.

### 5.2 I-4 — fixed the survivorship / look-ahead universe

**The problem:** `DJ30_TICKERS` was the *current* Dow-30 membership with
Sherwin-Williams (SHW) **back-filled to 2018**; SHW joined the DJIA only in Nov 2024.
Using 2024-knowledge constituents for 2019–2022 training is look-ahead /
survivorship bias.

**The fix — disclosed fixed neutral universe.** `config.UNIVERSE` is now the **24
names that were continuous DJIA members across the entire 2018-01 → 2025-01 window**
(NOT the live DJ-30). `config.EXCLUDED_FROM_DJ30` documents the 6 dropped names:

| Excluded | Reason |
|---|---|
| AMGN, CRM, HON | joined DJIA 2020-08-31 (not members during 2019–mid-2020 training) |
| DOW | joined 2019-04-02, removed 2024-11-08 (not continuous across window) |
| INTC | removed from DJIA 2024-11-08 (not a member through window end) |
| SHW | joined DJIA 2024-11-08 (back-filling to 2018 is look-ahead bias) |

**Retained (24):** AAPL, AXP, BA, CAT, CSCO, CVX, DIS, GS, HD, IBM, JNJ, JPM, KO,
MCD, MMM, MRK, MSFT, NKE, PG, TRV, UNH, V, VZ, WMT. `config.UNIVERSE` is the single
source of truth; `PortfolioEnv.UNIVERSE`, the backward-compat alias
`PortfolioEnv.DJ30_TICKERS`, and the `DJ30_TICKERS` in `data/pipeline.py` /
`data/sentiment_pipeline.py` all resolve to it. **State-dim is now 144**
(24 weights + 24 returns + 24×4 tech), down from 180 — update any doc that says 180.

### 5.3 Files changed in Phase 2

- **`config.py`** — added `UNIVERSE` (24) + `EXCLUDED_FROM_DJ30` (6, with reasons).
- **`environment/portfolio_env.py`** — `UNIVERSE` class attr from config;
  `DJ30_TICKERS` now an alias; default `self.tickers` uses `UNIVERSE`.
- **`data/pipeline.py`, `data/sentiment_pipeline.py`** — `DJ30_TICKERS ← config.UNIVERSE`.
- **`tuning/tune_runner.py`** — leak-free `_build_trial_envs`, `_val_net_sharpe`,
  shared `_run_one_trial`, thin `_train_trial`; **Ray-free `run_tune_simple` +
  `_sample_config`**; lazy Ray/HyperOpt imports; auto-fallback in `run_tune`;
  `_synthetic_df` spans train+val+test (24 names).
- **`test/test_no_leak.py`** — NEW (4 leak guards).
- **`tuning/best_config_LEAKED.json`** — NEW (archived pre-fix HPs).
- **`README.md`, `PHASE2_NOTES.md`** — Phase 2 disclosure + results + comparison.
- **`.gitignore`** — ignores run-artifact binaries.

### 5.4 Phase 2 commits (on `main`, authored Dhruvi, no co-author)

```
6a6238c  Phase 2: remove HPO + universe leakage (code, tests, docs)
0f2daea  Phase 2: leak-free HPO config + re-measured results
```

---

## 6. Current honest result (Phase 2: leak-free, 5 seeds × 500 ep)

Test window **2023-01-02 → 2025-01-30** (544 days), $1M start, 0.1% transaction +
0.1% slippage. Seeds `[0,1,2,3,4]`, 24-name universe, leak-free
`tuning/best_config.json`. Net of cost unless labelled "gross". (`make evaluate`.)

### 6.1 Headline — SAC agent (mean ± std, bootstrap 95% CI)

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| **Sharpe (net)** | **−0.081** | 0.743 | [−0.515, +0.657] |
| Sharpe (gross, pre-cost) | +1.489 | 0.155 | [+1.354, +1.625] |
| Sortino (net) | −0.101 | 1.115 | [−0.739, +1.012] |
| Calmar (net) | +0.160 | 0.762 | [−0.250, +0.923] |
| Max Drawdown | −19.8% | 5.6% | [−23.8%, −14.5%] |
| **Total Return** | **−2.3%** | 18.8% | [−13.2%, +16.5%] |
| Ann. Return (geom, net) | −1.5% | 8.3% | [−6.4%, +6.8%] |
| Ann. Volatility | 11.1% | 0.4% | [10.7%, 11.5%] |
| Win Rate | 48.9% | 1.8% | [47.5%, 50.4%] |
| Mean Turnover / step | 0.352 | 0.152 | [0.200, 0.437] |
| Mean HHI (1/N=0.042) | 0.092 | 0.019 | [0.073, 0.103] |
| Mean Active Share | 0.335 | 0.087 | [0.248, 0.382] |
| In-sample Sharpe (net) | −0.331 | 0.496 | — |

### 6.2 Baselines (deterministic; same costs)

| Strategy | Sharpe | Total Return |
|----------|--------|--------------|
| Equal Weight | **1.626** | +42.05% |
| SPY/QQQ 60/40 | 1.980 | +78.11% |
| Momentum 12-1 | 0.890 | +30.38% |
| Min Variance | 1.049 | +30.51% |
| Max Sharpe MVO | 1.463 | +62.91% |

### 6.3 Significance — SAC minus Equal-Weight Sharpe

- **PRIMARY (per-seed JK–Memmel):** **4/5 seeds significantly worse** at 5%
  (median p = 3.67e-10); ΔSharpe(annual) = **−1.706 ± 0.743**.
- **Deflated Sharpe Ratio** (n_trials=50): **DSR = 0.004** — effectively zero; the
  result does not survive a multiple-testing haircut.
- Cross-checks (optimistic; do not quote as headline): pooled JK z = −11.55, p ≈ 0;
  bootstrap 95% CI on ΔSharpe(annual) = [−2.044, −1.469] (excludes 0).

### 6.4 Phase 1 (leaky) vs Phase 2 (leak-free)

| | Phase 1 (leaky) | Phase 2 (leak-free) |
|---|---|---|
| Universe | 30 names, SHW back-filled | 24 continuous members |
| HPO objective | full series incl. test | validation-only, net of cost |
| **Net Sharpe** | +0.264 | **−0.081** |
| Gross Sharpe | +1.593 | +1.489 |
| Seeds worse than EW | 5/5 | 4/5 |
| **DSR** | 0.137 | **0.004** |

### 6.5 Interpretation (the finding)

The agent finds signal **gross** (Sharpe ≈ 1.49) but ~0.35/step turnover imposes a
**−1.57 Sharpe cost drag**, pulling **net** performance below zero and well below
equal-weight (1.63). Removing both leaks made the honest result *worse* and the DSR
≈ 0 — exactly as predicted (the gross edge shrank once HPO could no longer peek at
test). **Caveats to disclose:** (1) high seed variance (net Sharpe std 0.74 — why
4/5 not 5/5 reach significance); (2) the leak-free HPO config drove the entropy
temperature **α up to ≈6 by ep 500** (seed 0; the *inverse* of the prior collapse),
inflating turnover and widening the validation→test gap — a sign the 15-ep HPO
proxy transfers poorly to 500 ep (I-8) and the entropy mechanism needs work (I-6).
Both are out of Phase 2 scope (Phase 5).

---

## 7. The environment that works (authoritative)

Conda env **`portfolio-rl`**, Python **3.10**, macOS arm64, CPU. Top-level pins:

```
torch==2.2.2        numpy==1.26.4      pandas==2.2.2       scipy==1.13.1
gymnasium==0.29.1   yfinance==1.5.1    ray[tune]==2.9.3    hyperopt==0.2.7
matplotlib==3.8.4   seaborn==0.13.2    tensorboard==2.16.2 tqdm==4.66.4
ta==0.11.0          transformers==4.40.2
```

**Dependency landmines we hit (Phase 2):**
- **`pytest`** is a test-only dep, not in the runtime lock — `pip install pytest`
  before `make test`, then `make lock`.
- **Ray 2.9.3 is broken** against modern `setuptools` (≥81 removed `pkg_resources`;
  ≥71 removed `pkg_resources._vendor.packaging`) **and** modern `pyarrow` (removed
  `PyExtensionType`). The repo no longer depends on Ray working — the HPO falls back
  to a Ray-free random search automatically. If you *want* Ray Tune back, you'd need
  `pip install "setuptools<71"` **and** a compatible `pyarrow` (~12.x) — not
  recommended; the fallback is fine.
- `finrl`, `pyfolio-reloaded` intentionally **excluded**.

`make test` runs the full suite (**97 tests** = 93 + 4 Phase-2 no-leak guards).
After any clean install run `make lock`.

---

## 8. Repo map (key files)

```
main.py                      # CLI: train / tune / backtest (+ --seed)
config.py                    # date windows + UNIVERSE (24) + EXCLUDED_FROM_DJ30
agent/sac.py                 # SACAgent, DirichletActor, Critic, ReplayBuffer,
                             #   AssetTransformerEncoder; update() logs policy_entropy
environment/portfolio_env.py # Gym env; UNIVERSE class attr (alias DJ30_TICKERS);
                             #   state-dim 144; info.port_return is GROSS of cost
utils/trainer.py             # train(...seed=), backtest(...) — backtest is GROSS;
                             #   wrap to NET via diagnostics/_net_metrics
utils/metrics.py             # compute_sharpe/sortino/calmar/max_drawdown/all_metrics
utils/baselines.py           # equal_weight, spy_qqq, momentum_12_1, min_variance,
                             #   max_sharpe_mvo (value-path returns, net of cost)
utils/walk_forward.py        # expanding-window CV — EXISTS, NOT wired to CLI (Phase 3);
                             #   currently uses GROSS backtest() metrics — must wrap NET
utils/normalizer.py          # RunningNormalizer (Welford; train()/eval(); skips weights)
utils/significance.py        # JK–Memmel, bootstrap CI, PSR/DSR, PBO  (Phase 1)
utils/diagnostics.py         # turnover/HHI/active share, returns_from_values,
                             #   aggregate_metrics, bootstrap_ci  (Phase 1)
utils/plotting.py            # plot_diagnostics_panel, plot_alpha_entropy
experiments/multi_seed.py    # the honest single-window harness (make evaluate)
experiments/results/         # per_seed_results.csv, aggregate_metrics.json,
                             #   significance.json, baselines.json, run_meta.json,
                             #   per_seed_returns.npz, *.png, checkpoints/seed_*.pt
tuning/tune_runner.py        # LEAK-FREE HPO; Ray Tune w/ Ray-free fallback (Phase 2)
tuning/best_config.json      # leak-free tuned HPs (Phase 2)
tuning/best_config_LEAKED.json # archived pre-fix (leaky) HPs
data/pipeline.py             # download/indicators/splits; DJ30_TICKERS ← UNIVERSE
data/sentiment_pipeline.py   # FinBERT sentiment (optional; would raise state-dim)
test/                        # pytest suite incl. test_no_leak.py (Phase 2)
Makefile, environment.yml, requirements.lock.txt
PHASE2_NOTES.md              # Phase 2 design notes (companion to this doc)
PHASE1_HANDOVER.md, PHASE0_HANDOVER.md  # prior phase records
```

Useful signatures:
- `from config import UNIVERSE, EXCLUDED_FROM_DJ30, TRAIN_START, TRAIN_PROPER_END, VAL_START, TRAIN_END, TEST_START, TEST_END`
- `three_way_split(df, train_start, train_end, val_start, val_end, test_start, test_end) -> (train, val, test)`
- `backtest(agent, env, normalizer=None) -> dict` (GROSS — wrap with net!)
- `from utils.diagnostics import returns_from_values, policy_diagnostics, aggregate_metrics, bootstrap_ci`
- `from utils.significance import jobson_korkie_memmel, sharpe_diff_bootstrap_ci, deflated_sharpe_ratio`
- `from utils.walk_forward import walk_forward, walk_forward_summary`
  `walk_forward(df, agent_factory, normalizer_factory, n_folds=5, test_months=6, min_train_months=18, train_episodes=50, warmup_steps=500, env_kwargs=None) -> list[dict]`
  (each fold: expanding train up to a boundary, test on next `test_months`; returns
  per-fold metric dicts incl. `fold`, `train_start/end`, `test_start/end`).

---

## 9. PHASE 3 — Multi-regime walk-forward evaluation (DO THIS NEXT)

**Objective:** show whether the (now leak-free) agent is robust **across market
regimes**, not just on the single 2023–25 window. Wire `utils/walk_forward.py` into
a proper harness with **multi-seed per fold, net-of-cost metrics, CIs, and per-fold
significance vs equal-weight**, then report a per-regime table. Resolves the
regime part of **I-2**. **Budget:** code in ~1 day; the sweep is several hours
(folds × seeds) — reduce `train_episodes` per fold (e.g. 150–250) to keep it
overnight-able on the Mac.

### 9.1 Task A — fix `walk_forward` to report NET metrics

`walk_forward()` currently computes per-fold metrics via `trainer.backtest()`,
which returns **gross** Sharpe/Sortino/Calmar (the exact bug Phase 1 fixed for the
single-window harness). Before trusting any fold numbers, add a net path:
compute the fold's net-of-cost returns from `test_env.history["portfolio_value"]`
via `diagnostics.returns_from_values`, recompute Sharpe/Sortino/Calmar/total_return
on those, and report `gross_sharpe` alongside (mirror
`experiments/multi_seed._net_metrics`). This is **non-negotiable** — without it the
per-regime table will overstate performance.

### 9.2 Task B — build `experiments/walk_forward_eval.py` (the Phase 3 harness)

Mirror `experiments/multi_seed.py`’s structure. It should:
1. Load processed data (24-name universe) and pass the **full** chronological span
   (e.g. `TRAIN_START → TEST_END`, ~2019-04 → 2025-01) to `walk_forward` so folds
   sweep across 2020 (COVID), 2022 (bear), and 2023–24 (recovery). Walk-forward is
   leak-free **by construction** (expanding train, next-window test) — but still add
   an assertion per fold that `train_end < test_start`.
2. For each seed in `[0..N-1]`: call `walk_forward(...)` with
   `agent_factory`/`normalizer_factory` that build from `tuning/best_config.json`
   (reuse `main.build_agent`-style construction, encoder default `mlp`). Seed each
   seed with `set_global_seed(seed)`.
3. For each fold: compute the **net** agent metrics (Task A), an **equal-weight
   baseline over the same fold test window** (`utils.baselines.equal_weight`), the
   per-fold **JK–Memmel** ΔSharpe vs that baseline, plus turnover/HHI/active-share
   via `policy_diagnostics`.
4. **Aggregate two ways:** (a) **per-regime** — group folds by calendar regime and
   report mean ± std + bootstrap CI across seeds within each regime; (b) **overall**
   — pooled across folds/seeds. Reuse `diagnostics.aggregate_metrics` and
   `diagnostics.bootstrap_ci`.
5. Persist `walk_forward_per_fold.csv` (seed × fold rows), `walk_forward_regime.json`
   (per-regime table), `walk_forward_significance.json`, a `run_meta.json` (via
   `write_run_meta`), and a figure (per-regime net Sharpe with CIs; reuse/extend
   `utils.plotting`).
6. Add `make walkforward` to the Makefile (`SEEDS`, `FOLDS`, `TEST_MONTHS`,
   `EPISODES` overridable), and optionally a `python main.py --mode walkforward`
   path.

### 9.3 Task C — regime labelling

Map each fold’s `test_start/test_end` to a regime label and print a per-regime
table: e.g. **COVID crash/recovery (2020-02 → 2020-12)**, **2021 bull**,
**2022 bear (rate shock)**, **2023–24 recovery/AI rally**. Document the exact date
cutoffs you choose. Report, per regime: net Sharpe (mean ± CI across seeds), net
total return, turnover, and `k/N folds beating equal-weight`. State plainly where
the agent does best/worst — the expectation from §6 is it underperforms EW net in
most regimes, but disclose any regime where it’s competitive.

### 9.4 Task D — unit tests (`test/test_walk_forward.py`)

- Folds are **chronological and non-overlapping**, training window **expands**, and
  every fold has `train_end < test_start` (no future leak).
- `walk_forward` stops cleanly when a fold would exceed the data window.
- The **net** metric path is used (e.g. assert the harness’s fold Sharpe differs
  from the gross `backtest()` Sharpe when turnover > 0, or assert the net key
  exists and is derived from the value path).
- Synthetic-data only (no downloads, no Ray), so it runs in CI within the 97-test
  suite.

### 9.5 Acceptance criteria (Phase 3)

- `utils/walk_forward.py` reports **net-of-cost** metrics (gross exposed alongside).
- Walk-forward harness runs **N≥3 seeds × the available folds** across ≥3 regimes;
  per-regime table + overall table, each with mean ± std and bootstrap 95% CI.
- Per-fold (or per-regime) **JK–Memmel** ΔSharpe vs equal-weight, with the same
  honest framing (per-seed primary; pooled/bootstrap as cross-checks).
- Artifacts persisted (`per_fold.csv`, `regime.json`, `significance.json`,
  `run_meta.json`, figure); `make walkforward` works; new unit tests pass.
- No leakage (assert `train_end < test_start` per fold); authored Dhruvi, no co-author.

### 9.6 Suggested commands (zsh-safe — no inline comments)

```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization
pip install pytest >/dev/null 2>&1 ; make test
python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 8 --test-months 6 --episodes 200 --config tuning/best_config.json
git add experiments/walk_forward_eval.py utils/walk_forward.py test/test_walk_forward.py Makefile experiments/results/walk_forward_*.json experiments/results/walk_forward_*.csv
git commit -m "Phase 3: multi-regime walk-forward evaluation"
```

### 9.7 Explicitly OUT of scope for Phase 3

- Entropy/reward fixes (I-6, I-10, the α blow-up) → **Phase 5**.
- New baselines (SPY buy-and-hold, 60/40 SPY/AGG, risk parity, Ledoit-Wolf MVO)
  (I-7) → **Phase 4**.
- README/doc rewrite, state-dim/FinRL fixes (I-9) → **Phase 6**.
- Re-tuning HPO with longer proxies (I-8 transfer gap) → **Phase 5/6**.

---

## 10. Roadmap beyond Phase 3 (context, not action)

- **Phase 4** — baseline overhaul: SPY buy-and-hold, 60/40 SPY/AGG, risk parity,
  rolling re-estimated MVO with Ledoit–Wolf shrinkage; same costs, seeds, CIs.
- **Phase 5** — entropy/learning-signal: **explicitly fix the entropy temperature
  pathology** (collapse → blow-up depending on config), entropy floor / scheduled α
  / re-derived target entropy; reward-scaling sweep; **target the turnover problem
  from §6** (turnover penalty / cost-aware reward) since costs are the binding
  constraint; ablate MLP vs transformer and ± FinBERT. Also fix the **reproducibility
  nondeterminism** (§12) and consider a longer-episode HPO proxy (I-8).
- **Phase 6** — sync docs to code (state-dim **144**, "FinRL-compatible" not
  -powered, repo tree), rewrite results around the leak-free CI-backed numbers, add
  a Limitations section + one-command reproduction recipe.

Then a **Capstone Roadmap** (only after results are trustworthy): Docker, W&B +
Hydra tracking, GitHub Actions CI, a deployed Streamlit/Gradio demo, algorithm
breadth (PPO/TD3 + CVaR risk-aware reward), and a GenAI extension (LLM
news/filings → signals, RAG over 10-Ks, LLM rebalance rationales).

---

## 11. Definition of Done (whole remediation effort)

- ✅ All runs seeded, version-pinned, commit+config stamped (Phase 0).
- ✅ Every headline metric = mean ± 95% CI across ≥5 seeds, named significance test
  vs equal-weight (Phase 1).
- ✅ No leakage in HPO or universe; both unit-test-guarded and documented (Phase 2).
- ⬜ Results across ≥3 market regimes (Phase 3).
- ⬜ Strong baseline set (Phase 4).
- ◑ Policy behavior explained (✅ Phase 1); entropy mechanism functioning or no
  longer claimed (Phase 5).
- ⬜ README matches code, has a Limitations section + one-command repro (Phase 6).

---

## 12. Gotchas / lessons carried forward

- **Always report NET (value-path) Sharpe.** `trainer.backtest()` returns **gross**
  metrics; `multi_seed._net_metrics()` wraps it. **`walk_forward.py` still returns
  gross — fix it first in Phase 3 (Task A).** This exact bug produced an
  inconsistent "Sharpe +1.3 next to total_return −7%" table once.
- **Per-seed JK is primary**; pooled/bootstrap are optimistic cross-checks.
- **zsh does not treat `#` as a comment.** Inline `# comments` on command lines get
  passed as arguments and break the command (this happened twice). Give comment-free
  command blocks.
- **Ray is broken in this env and you should not fight it.** setuptools ≥71 removed
  `pkg_resources._vendor.packaging`; pyarrow removed `PyExtensionType`. The HPO
  already auto-falls-back to a Ray-free random search with the same leak-free
  objective. Don’t downgrade a pile of packages to revive Ray.
- **Reproducibility nondeterminism (open issue).** Two `make evaluate` runs with
  identical inputs gave different per-seed results (seed 0 OOS Sharpe 0.17 vs 0.40).
  `set_global_seed` is not giving bitwise determinism on this Mac — likely
  nondeterministic CPU torch ops or an unseeded RNG path (env `reset()` /
  `action_space.sample()` / buffer). Worth hardening early in Phase 3 (it widens the
  walk-forward variance bands): try `torch.use_deterministic_algorithms(True)`,
  fix thread count (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), and audit
  every RNG source in the train loop.
- **Entropy α can blow up, not just collapse.** With the leak-free config seed 0’s
  α reached ≈6 by ep 500 (`final_alpha` in `per_seed_results.csv`), raising
  turnover. The entropy mechanism is unreliable across configs — Phase 5.
- **Don’t `git add -A`** — it commits large `.pt`/`.npz`/`.pkl` binaries. `.gitignore`
  now covers them; add source + small summaries explicitly.
- A stray `diff --git a/checkpoints/best_agent.pt …` line sometimes prints during a
  run — benign terminal/git noise, not emitted by the harness; ignore it.
- `pytest` is not in the runtime lock; install it before `make test`.
- If git refuses with a stale `.git/index.lock`, remove that file and retry.
- **State-dim is 144 now** (24 assets), not 180/210 — older docs are stale (I-9).
