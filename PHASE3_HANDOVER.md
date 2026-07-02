# SAC Portfolio Optimization — Phase 3 Handover & Phase 4 Brief

**Paste this whole document into a fresh Claude chat to continue the project.**
It contains everything needed: project context, what Phases 0–3 changed and why,
the exact environment that works (including the dependency landmines), the current
honest leak-free multi-regime results, and a complete, actionable specification
for Phase 4 (baseline overhaul).

- **Repo:** https://github.com/dhruvi002/dynamic-portfolio-optimization
- **Local path (Mac):** `~/Documents/Projects/finrl/dynamic-portfolio-optimization`
- **Conda env:** `portfolio-rl` (Python 3.10, macOS arm64, CPU-only)
- **Phase 0–2:** June 2026 · **Phase 3:** July 2026
- **HEAD at handover:** `5d62780` (Phase 3 complete)
- **Owner:** Dhruvi (MS-CS student targeting entry-level Data / ML / Applied-Research / GenAI roles)

---

## 0. Standing instructions for the assistant (read first)

1. **Honest reporting over big numbers.** Every headline metric ships with a
   confidence interval and a significance test. Never re-introduce a bare
   "+24%"/"+47%" headline. If an edge is not statistically significant, say so.
   Report **net-of-cost** metrics — `utils/trainer.backtest()` returns **gross**
   metrics; wrap them with the net value-path computation (this has bitten the
   project three times now; see §12).
2. **Reproducibility before results.** Runs are seeded, version-pinned, and
   stamped (`run_meta.json`). Keep it intact. NOTE: there is a known residual
   nondeterminism on this Mac (§12) still worth fixing (deferred to Phase 5).
3. **No leakage.** Phase 2 removed an HPO leak and a survivorship/look-ahead
   universe leak; Phase 3's walk-forward is leak-free *by construction* (expanding
   train, next-window test, `train_end < test_start` asserted per fold). All three
   are unit-test-guarded (`test/test_no_leak.py`, `test/test_walk_forward.py`).
   **Any new baseline or evaluation in Phase 4 must preserve this** — rolling
   estimators may use only data up to each rebalance date.
4. **CPU-first.** Model is small (MLP, 24 assets, state-dim 144). Phase 3's
   walk-forward (9 folds × 3 seeds × 200 ep) took roughly **6–9 h** wall-clock on
   a MacBook Air because the expanding training window grows each fold. Phase 4 is
   mostly *analytic* baselines (cheap) but re-runs the harnesses — budget a few
   hours if you re-train the agent, near-zero if you reuse cached agent returns.
5. **`caffeinate` all long runs.** Wrap any multi-hour command in `caffeinate -i`
   so the Mac doesn't sleep mid-run (a Phase-3 fold stalled 2h25m from sleep). e.g.
   `caffeinate -i python experiments/walk_forward_eval.py …`.
6. **Git: do NOT add Claude / Sonnet / Opus as a co-author.** Commit regularly,
   authored as the user only. No `Co-authored-by` trailers. Do **not** `git add -A`
   blindly — it sweeps in large `.pt`/`.npz`/`.pkl` binaries; `.gitignore` covers
   `checkpoints/ runs/ plots/ experiments/results/checkpoints/ *.pt *.pkl *.npz`.
   Commit source + the small JSON/CSV/PNG summaries explicitly. If git refuses with
   a stale `.git/index.lock`, remove that file and retry.
7. **Their shell is zsh, which does NOT treat `#` as a comment by default** — put
   no inline `# comments` on command lines (they get passed as arguments and break
   the command). Keep command blocks comment-free.
8. **Use the existing harnesses.** `experiments/multi_seed.py` (`make evaluate`) is
   the single-window instrument; `experiments/walk_forward_eval.py`
   (`make walkforward`) is the multi-regime instrument. Reuse their
   aggregation/significance helpers, don't reinvent them.

---

## 1. Project in one paragraph

A deep reinforcement-learning portfolio optimizer for a fixed large-cap US equity
universe. A **Soft Actor-Critic (SAC)** agent with a **Dirichlet policy on the
K-simplex** allocates weights across 24 continuous-membership DJIA names. The
environment is a hand-rolled Gymnasium env (FinRL-*compatible*, not FinRL-powered).
The codebase is strong ML engineering — twin critics, a Welford running normalizer
that freezes at eval, chronological train/val/test split, checkpoint selection on
validation Sharpe, transaction + slippage costs applied identically to agent and
baselines, an optional cross-asset transformer encoder, and an optional FinBERT
sentiment channel. There is a leak-free HPO runner (Ray-free fallback), a Phase-1
single-window honest-evaluation harness, and now a Phase-3 **multi-regime
walk-forward** harness. **The original headline ("+24% Sharpe / +47.3% return over
equal-weight") rested on a single unseeded, leaky backtest.** Phases 0–3 replaced
it with a reproducible, multi-seed, CI-backed, significance-tested, leakage-free,
**multi-regime** evaluation. **The honest verdict, now confirmed across four market
regimes: after costs the agent does not beat — it loses to — equal-weight in every
one of 27 folds (overall net Sharpe −0.44 vs EW; DSR 0.004). The binding
constraint is turnover/transaction cost, not signal (gross Sharpe is +1.15).**

---

## 2. Issue register (full remediation backlog)

Severity: **H** = blocks the headline claim, **M** = materially weakens credibility, **L** = polish.

| ID | Issue | Sev | Phase | Status |
|----|-------|-----|-------|--------|
| I-1 | No global seeding → not reproducible | H | 0 | ✅ DONE |
| I-2 | Single backtest/seed/regime; no significance/CI | H | 1, 3 | ✅ **DONE (regimes closed in Phase 3)** |
| I-3 | HPO leaks the test set | H | 2 | ✅ DONE |
| I-4 | Survivorship / look-ahead universe (SHW back-filled) | H | 2 | ✅ DONE |
| I-5 | IS-vs-OOS gap unexplained; no turnover/concentration/active-share | H | 1 | ✅ DONE |
| I-6 | Entropy temperature pathology (collapse ↔ blow-up) | M | 5 | ◑ MEASURED (fix → Phase 5) |
| I-7 | Weak/strawman baselines; SPY buy-and-hold missing | M | **4** | ⬜ **TODO — DO THIS NEXT** |
| I-8 | Config ambiguity; 15-ep HPO proxy may not transfer to 500-ep | M | 5, 6 | ◑ PARTIAL (leak-free; transfer gap remains) |
| I-9 | Doc/code mismatches (state-dim **144**, FinRL claim, tree) | L | 6 | ⬜ TODO |
| I-10 | Tiny reward signal (1e-4 scaling); learning-signal not validated | M | 5 | ⬜ TODO |
| I-11 | Headline overstates precision | M | 1 | ✅ DONE |

Dependencies: **0** → **1** → **2** (leak removal) → **3** (regimes — done) →
**4** (baselines — next) → **5** (entropy/reward/turnover) → **6** (docs + honest claims).

---

## 3. What Phases 0–2 did (context)

- **Phase 0** — determinism + self-documentation: `utils/seeding.py`
  (`set_global_seed`), `utils/run_meta.py` (`run_meta.json` with UTC time, git SHA
  +`-dirty`, seed, config, device, versions), `environment.yml`,
  `requirements.lock.txt`, a `Makefile`, `--seed` wired through. Removed broken deps
  (`finrl`, `pyfolio-reloaded`), added `scipy`, upgraded `yfinance`.
- **Phase 1** — statistically defensible single-window evaluation: multiple seeds,
  bootstrap CIs, Jobson–Korkie–Memmel Sharpe-difference test vs equal-weight,
  overfitting-aware Sharpe (PSR/DSR/PBO), policy diagnostics (turnover/HHI/active
  share), IS-vs-OOS. New: `utils/significance.py`, `utils/diagnostics.py`,
  `experiments/multi_seed.py` (`make evaluate`), 22 analytic unit tests.
  **Convention:** test statistics use *per-period* Sharpe; annualization is
  display-only. **Primary significance = per-seed JK–Memmel** (each seed a
  full-length OOS record); pooled/bootstrap are optimistic cross-checks only.
- **Phase 2** — made the pipeline **leakage-free**: killed the HPO test-set leak
  (`tuning/tune_runner.py` trains train-only, scores net-of-cost validation-only,
  hard leak guard, Ray-free random-search fallback), and fixed the
  survivorship/look-ahead universe. `config.UNIVERSE` is now the **24 continuous
  DJIA members 2018-01 → 2025-01** (NOT the live DJ-30); `config.EXCLUDED_FROM_DJ30`
  documents the 6 dropped names (AMGN/CRM/HON joined 2020-08; DOW/INTC/SHW mid-sample).
  **State-dim is 144** (24 weights + 24 returns + 24×4 tech). Guards in
  `test/test_no_leak.py`.

**Phase-2 single-window honest result** (test 2023-01 → 2025-01, 5 seeds × 500 ep):
net Sharpe **−0.081** vs equal-weight **+1.626**; gross Sharpe +1.489; **DSR 0.004**;
4/5 seeds significantly worse than EW.

---

## 4. What Phase 3 did (THIS WAS THE NEW WORK)

**Objective:** show whether the (leak-free) agent is robust **across market
regimes**, not just on the single 2023–25 window. Wired `utils/walk_forward.py`
into a proper harness with multi-seed per fold, net-of-cost metrics, per-regime
CIs, and per-fold significance vs equal-weight. **Resolved the regime part of
I-2.** Full design notes: `PHASE3_NOTES.md`.

### 4.1 Task A — `utils/walk_forward.py` now reports NET-of-cost metrics

**The bug (before):** `walk_forward()` computed per-fold metrics via
`trainer.backtest()`, which returns **gross** Sharpe/Sortino/Calmar (exactly the
bug Phase 1 fixed for the single-window harness). The fix:

- New `_fold_net_metrics(test_env)` recomputes Sharpe/Sortino/Calmar/total_return
  from the **net-of-cost value path** (`diagnostics.returns_from_values`) and
  exposes `gross_sharpe` alongside (mirrors `multi_seed._net_metrics`). **Key
  `sharpe` is now NET.**
- Per-fold policy diagnostics (turnover / HHI / active-share) via
  `diagnostics.policy_diagnostics`.
- Each fold attaches its raw net daily return series (`agent_returns`) + aligned
  `test_dates` so the harness can run a per-fold JK test.
- **Leak assertion `train_end < test_start` per fold** (redundant with the
  construction, kept explicit).
- Fold scheduling factored into a pure, unit-testable `_fold_windows(...)`
  (chronology / non-overlap / expanding-train / clean-stop).

### 4.2 Task B — `experiments/walk_forward_eval.py` (the Phase 3 harness)

Mirrors `experiments/multi_seed.py`. For each seed: `set_global_seed(seed)`, run
`walk_forward(...)` over the **full** span `[TRAIN_START, TEST_END]` (≈2019-04 →
2025-01) built from `tuning/best_config.json`, encoder `mlp`. For each fold:
compute NET agent metrics, an **equal-weight baseline over the same fold test
window**, the per-fold **JK–Memmel** ΔSharpe vs that baseline, and turnover/HHI/
active-share. **Aggregates two ways:** (a) **per-regime** (mean ± std + bootstrap
95% CI across folds/seeds in each regime); (b) **overall** (pooled). Plus a
**Deflated Sharpe Ratio** as a multiple-testing cross-check. Reuses
`diagnostics.aggregate_metrics` / `bootstrap_ci` and `significance.*`.

### 4.3 Task C — regime labelling (`utils/regimes.py`)

Each fold is assigned to a regime by its test-window **midpoint**. Documented,
fixed cutoffs (inclusive):

| Regime | Dates |
|---|---|
| pre-COVID 2019 | 2019-01-01 → 2020-01-31 |
| COVID crash/recovery | 2020-02-01 → 2020-12-31 |
| 2021 bull | 2021-01-01 → 2021-12-31 |
| 2022 bear (rate shock) | 2022-01-01 → 2022-12-31 |
| 2023-24 recovery/AI | 2023-01-01 → 2025-12-31 |

Defaults `min_train_months=12, test_months=6, folds=9` give 9 folds sweeping all
four active regimes with ≥2 folds each (COVID ×2, 2021 bull ×2, 2022 bear ×2,
recovery ×3). `min_train_months=12` (first test ≈2020-04) is the deliberate choice
that buys COVID coverage.

### 4.4 Task D — unit tests (`test/test_walk_forward.py`)

Synthetic-data only (no downloads, no Ray, no SAC training): fold chronology /
non-overlap / expanding-train / `train_end<test_start`; clean stop past the data
window; the **net** metric path differs from the gross `backtest()` Sharpe when
turnover>0 and is derived from the value path; regime labelling. **Suite is now
106 tests (was 97).**

### 4.5 Wiring

`make walkforward` (overridable `WF_SEEDS FOLDS TEST_MONTHS MIN_TRAIN_MONTHS
WF_EPISODES`) and `python main.py --mode walkforward`. Per-regime Sharpe figure via
new `utils.plotting.plot_walk_forward_regimes`.

### 4.6 Files changed in Phase 3

- **`utils/walk_forward.py`** — NET metrics, `_fold_windows`, `_fold_net_metrics`,
  `_agent_net_returns`, per-fold diagnostics + leak assertion (rewritten).
- **`utils/regimes.py`** — NEW (regime bands + `regime_for` + `regimes_present`).
- **`experiments/walk_forward_eval.py`** — NEW (the Phase 3 harness;
  `run_walk_forward_eval(...)` is importable by `main.py`).
- **`utils/plotting.py`** — added `plot_walk_forward_regimes`.
- **`test/test_walk_forward.py`** — NEW (9 guards).
- **`Makefile`** — `walkforward` target + `WF_*` vars.
- **`main.py`** — `--mode walkforward` + `--wf-seeds/--folds/--test-months/--min-train-months`.
- **`PHASE3_NOTES.md`** — NEW (design notes + results table).

### 4.7 Phase 3 commit (on `main`, authored Dhruvi, no co-author)

```
5d62780  Phase 3: multi-regime walk-forward evaluation
```

Committed source + small summaries: `experiments/results/walk_forward_per_fold.csv`,
`walk_forward_regime.json`, `walk_forward_significance.json`,
`walk_forward_regimes.png`, `run_meta.json`.

---

## 5. Current honest result (Phase 3: leak-free, multi-regime, 3 seeds × 9 folds × 200 ep)

Full span 2019-04 → 2025-01, $1M start, 0.1% transaction + 0.1% slippage. Seeds
`[0,1,2]`, 24-name universe, leak-free `tuning/best_config.json`. Net of cost
unless labelled "gross". (`make walkforward`.)

### 5.1 Per-regime — agent NET Sharpe vs Equal-Weight

| Regime | Agent NET Sharpe (mean ± std, 95% CI) | EW Sharpe | k/N > EW | JK sig @5% | median p | ΔSharpe(ann) vs EW |
|---|---|---|---|---|---|---|
| COVID crash/recovery | **+1.294** ± 0.422 [+0.96, +1.63] | +2.387 | 0/6 | 5/6 | 6.1e-3 | −1.093 ± 0.547 |
| 2021 bull | −0.972 ± 0.403 [−1.28, −0.64] | +0.708 | 0/6 | 6/6 | 3.2e-6 | −1.680 ± 0.394 |
| 2022 bear (rate shock) | −1.410 ± 1.465 [−2.44, −0.25] | +0.063 | 0/6 | 6/6 | 1.7e-4 | −1.473 ± 0.491 |
| 2023-24 recovery/AI | −0.590 ± 1.320 [−1.45, +0.26] | +1.920 | 0/9 | 8/9 | 4.8e-6 | −2.510 ± 0.753 |

### 5.2 Overall (pooled across 27 folds)

| Metric | Value |
|---|---|
| **NET Sharpe** | **−0.438** ± 1.443  [−0.982, +0.103] |
| Gross Sharpe (pre-cost) | +1.145 |
| Folds beating EW | **0 / 27** |
| Per-fold JK significant @5% | 25 / 27 (median p ≈ 1.7e-4) |
| ΔSharpe(annual) vs EW | **−1.780 ± 0.806** |
| **Deflated Sharpe Ratio** (n_trials=50) | **0.0045** ≈ 0 |

### 5.3 Interpretation (the finding)

The Phase-2 single-window verdict **generalizes across regimes**: net of cost the
agent loses to equal-weight in **every** fold. The ~1.58 **gross→net collapse**
(+1.15 → −0.44) is the transaction-cost drag from ~0.4/step turnover — the same
mechanism identified in Phase 2 §6, now shown to hold in COVID, the 2021 bull, the
2022 bear, and the 2023–24 recovery. The agent's *only* positive-Sharpe regime is
COVID (+1.29), but even there equal-weight is higher (+2.39), so it is **not
competitive anywhere net of cost**. DSR ≈ 0 means the result does not survive a
multiple-testing haircut. **The binding constraint is turnover, not signal** — this
is the single most important handoff into Phase 5 (cost-aware reward / turnover
penalty). Caveat: wide bands in 2022 (net Sharpe std 1.47) reflect the residual
reproducibility nondeterminism (§12) plus genuine regime difficulty.

---

## 6. The environment that works (authoritative)

Conda env **`portfolio-rl`**, Python **3.10**, macOS arm64, CPU. Top-level pins
(from the Phase-3 `run_meta.json`):

```
torch==2.2.2        numpy==1.26.4      pandas==2.2.2       scipy==1.13.1
gymnasium==0.29.1   yfinance==1.5.1    ray[tune]==2.9.3    hyperopt==0.2.7
matplotlib==3.8.4   seaborn==0.13.2    tensorboard==2.16.2 tqdm==4.66.4
ta==0.11.0          transformers==4.40.2
```

**scikit-learn is NOT currently installed** (`run_meta` shows `scikit-learn: null`).
**Phase 4 needs it** for Ledoit–Wolf shrinkage (`sklearn.covariance.LedoitWolf`) —
`pip install scikit-learn` then `make lock` (see §7 below).

**Dependency landmines:**
- **`pytest`** is a test-only dep, not in the runtime lock — `pip install pytest`
  before `make test`, then `make lock`.
- **Ray 2.9.3 is broken** against modern setuptools/pyarrow. The HPO auto-falls-back
  to a Ray-free random search with the identical leak-free objective. Don't fight it.
- `finrl`, `pyfolio-reloaded` intentionally **excluded**.
- **CPU torch wheels are proxy-restricted in some sandboxes** — install on the Mac.

`make test` runs the full suite (**106 tests** = 97 pre-Phase-3 + 9 walk-forward
guards). After any clean install run `make lock`.

---

## 7. Repo map (key files)

```
main.py                      # CLI: train / tune / backtest / walkforward (+ --seed)
config.py                    # date windows + UNIVERSE (24) + EXCLUDED_FROM_DJ30
agent/sac.py                 # SACAgent, DirichletActor, Critic, ReplayBuffer,
                             #   AssetTransformerEncoder; update() logs policy_entropy
environment/portfolio_env.py # Gym env; UNIVERSE class attr (alias DJ30_TICKERS);
                             #   state-dim 144; info.port_return is GROSS of cost
utils/trainer.py             # train(...seed=), backtest(...) — backtest is GROSS
utils/metrics.py             # compute_sharpe/sortino/calmar/max_drawdown/all_metrics
utils/baselines.py           # equal_weight, spy_qqq, momentum_12_1, min_variance,
                             #   max_sharpe_mvo (value-path returns, net of cost)
                             #   ← PHASE 4 EXTENDS THIS FILE
utils/walk_forward.py        # Phase-3: NET metrics, _fold_windows, _fold_net_metrics,
                             #   per-fold diagnostics + leak assertion
utils/regimes.py             # Phase-3: regime bands + regime_for + regimes_present
utils/normalizer.py          # RunningNormalizer (Welford; train()/eval(); skips weights)
utils/significance.py        # JK–Memmel, bootstrap CI, PSR/DSR, PBO  (Phase 1)
utils/diagnostics.py         # turnover/HHI/active share, returns_from_values,
                             #   aggregate_metrics, bootstrap_ci  (Phase 1)
utils/plotting.py            # + plot_walk_forward_regimes  (Phase 3)
experiments/multi_seed.py    # single-window honest harness (make evaluate)
experiments/walk_forward_eval.py # Phase-3 multi-regime harness (make walkforward);
                             #   run_walk_forward_eval(...) importable by main.py
experiments/results/         # per_seed_results.csv, aggregate_metrics.json, ...,
                             #   walk_forward_per_fold.csv, walk_forward_regime.json,
                             #   walk_forward_significance.json, walk_forward_regimes.png
tuning/tune_runner.py        # LEAK-FREE HPO; Ray Tune w/ Ray-free fallback (Phase 2)
tuning/best_config.json      # leak-free tuned HPs (Phase 2)
data/pipeline.py             # download/indicators/splits; DJ30_TICKERS ← UNIVERSE
data/sentiment_pipeline.py   # FinBERT sentiment (optional; would raise state-dim)
test/                        # pytest suite incl. test_no_leak.py, test_walk_forward.py
Makefile, environment.yml, requirements.lock.txt
PHASE3_NOTES.md              # Phase 3 design notes + results (companion to this doc)
PHASE0/1/2_HANDOVER.md, PHASE1/2_NOTES.md  # prior phase records
```

Useful signatures:
- `from config import UNIVERSE, EXCLUDED_FROM_DJ30, TRAIN_START, TRAIN_PROPER_END, VAL_START, TRAIN_END, TEST_START, TEST_END`
- `three_way_split(df, train_start, train_end, val_start, val_end, test_start, test_end) -> (train, val, test)`
- `backtest(agent, env, normalizer=None) -> dict` (GROSS — wrap with net!)
- `from utils.diagnostics import returns_from_values, policy_diagnostics, aggregate_metrics, bootstrap_ci`
- `from utils.significance import jobson_korkie_memmel, sharpe_diff_bootstrap_ci, deflated_sharpe_ratio`
- `from utils.walk_forward import walk_forward, _fold_windows, _fold_net_metrics`
- `from utils.regimes import regime_for, regimes_present, REGIME_ORDER`
- `from utils.baselines import equal_weight, spy_qqq, momentum_12_1, min_variance, max_sharpe_mvo`
  (all return `(metrics_dict, portfolio_values, dates)`, monthly rebalance,
  identical cost model: `cost = (tc+slip)·Σ|Δw|·value`)

---

## 8. PHASE 4 — Baseline overhaul (DO THIS NEXT)

**Objective:** replace the weak/strawman baseline set with a **strong, standard**
one so the agent's net underperformance is measured against benchmarks a
practitioner would actually respect. Resolves **I-7**. This is mostly *analytic*
(no RL training) and therefore cheap; the expensive part is only if you re-run the
agent, which you can avoid by reusing cached per-seed/per-fold agent returns.

**The problem.** Current `utils/baselines.py` has: `equal_weight`, `spy_qqq`
(60/40 SPY/QQQ — two tech-heavy equity ETFs, not a real balanced benchmark),
`momentum_12_1`, `min_variance` and `max_sharpe_mvo` (both estimated **once** on
the full train window → static weights, mild look-ahead-ish for walk-forward and
not how MVO is run in practice). **SPY buy-and-hold — the single most obvious
market benchmark — is missing.** The significance tests currently compare the
agent only vs **equal-weight**.

### 8.1 Task A — add the missing standard baselines (`utils/baselines.py`)

Add these, each with the **same cost model, monthly rebalance, and
`(metrics, values, dates)` return signature** as the existing functions, and each
**net of cost**:

1. **`spy_buy_and_hold(start, end, ...)`** — single-asset SPY, buy once, no
   rebalancing (turnover cost only at t0). The canonical market benchmark. Download
   SPY via yfinance (cache).
2. **`spy_agg_60_40(start, end, ...)`** — 60% SPY / 40% **AGG** (US aggregate
   bonds), monthly rebalance. The classic balanced benchmark (this is what "60/40"
   means; keep `spy_qqq` too but it is NOT the 60/40 benchmark). Download SPY+AGG.
3. **`risk_parity(df, tickers, train_df=None, ...)`** — inverse-volatility (or
   equal-risk-contribution) weights over the 24-name universe, estimated from data
   **available up to each rebalance date** (trailing window, e.g. 252 trading days),
   rebalanced monthly. No look-ahead.
4. **`rolling_mvo_ledoit_wolf(df, tickers, kind="min_var"|"max_sharpe", lookback=252, ...)`**
   — mean-variance optimization with **Ledoit–Wolf shrinkage**
   (`sklearn.covariance.LedoitWolf`), **re-estimated at each rebalance** on a
   trailing `lookback` window (rolling, not static). Long-only, weights sum to 1.
   Reuse the SLSQP solvers already in `baselines.py` (`_min_variance_weights`,
   `_max_sharpe_weights`) but feed them the shrunk covariance / rolling mean. This
   is the "MVO done properly" benchmark that replaces the static `min_variance` /
   `max_sharpe_mvo` (keep the old ones for backward-compat + as a "static MVO"
   contrast, but the rolling LW versions are the headline).

**Add `scikit-learn` to the env:** `pip install scikit-learn`, add to
`environment.yml` + `requirements.txt`, then `make lock`. Guard the import so the
module still imports if sklearn is absent (fall back to sample covariance with a
printed warning), mirroring the lazy-import pattern used for Ray.

### 8.2 Task B — wire the new baselines into BOTH harnesses

1. **`main._collect_baselines(test_df, train_df)`** — add SPY B&H, 60/40 SPY/AGG,
   risk parity, rolling-LW min-var and max-Sharpe. Keep existing ones. This feeds
   `main.py` train/backtest and `experiments/multi_seed.py` (single-window).
2. **`experiments/multi_seed.py`** — its baseline point-estimates come from
   `_collect_baselines`; additionally run the **per-seed JK–Memmel of the agent vs
   each baseline** (currently only vs equal-weight) so the single-window table
   reports significance vs the *strongest* benchmark, not just EW. Add a
   `baseline_returns` alignment helper mirroring `_equal_weight_return_series`.
3. **`experiments/walk_forward_eval.py`** — for each fold, in addition to the
   equal-weight baseline, compute each new baseline **over the same fold test
   window using only data up to `test_start`** (pass `train_df = df[date < test_start]`
   for the estimators), and run per-fold JK vs each. Extend
   `walk_forward_per_fold.csv` and the per-regime aggregation to carry, per
   baseline: its Sharpe, ΔSharpe vs agent, and `k/N folds the agent beats it`.
   **Report the agent vs the BEST baseline per regime as the headline comparison.**

### 8.3 Task C — extend the figure + tables

Extend `utils.plotting.plot_walk_forward_regimes` (or add a companion) to show the
agent against the **full baseline panel** per regime (grouped bars with CIs), and
print a per-regime table of `agent NET Sharpe` vs each baseline with `k/N > baseline`.
Keep the honest framing (per-fold/per-seed JK primary; DSR cross-check).

### 8.4 Task D — unit tests (`test/test_baselines.py` extend, or new file)

- **Cost consistency:** every new baseline applies `(tc+slip)·Σ|Δw|·value` exactly
  like `PortfolioEnv` (assert against a hand-computed 1-rebalance example).
- **No look-ahead:** risk-parity and rolling-MVO weights at rebalance date *t* are a
  function of data with `date ≤ t` only (assert changing future rows leaves weights
  at *t* unchanged).
- **Well-formed weights:** long-only, sum to 1 (± tol), correct length.
- **Ledoit–Wolf:** shrinks toward a scaled identity (assert off-diagonal magnitude
  shrinks vs sample covariance on a noisy synthetic panel); graceful fallback when
  sklearn is absent.
- **SPY B&H:** turnover cost incurred once (at t0), zero thereafter.
- Synthetic data / mocked yfinance where possible so tests run in CI within the
  (now 106+) suite without live downloads.

### 8.5 Acceptance criteria (Phase 4)

- `utils/baselines.py` provides SPY buy-and-hold, 60/40 SPY/AGG, risk parity, and
  rolling Ledoit–Wolf MVO (min-var + max-Sharpe), all net of cost, same cost model,
  no look-ahead.
- Both harnesses compare the agent against the **full** baseline set with the same
  seeds/CIs, and per-fold/per-seed **JK–Memmel** vs each baseline (headline =
  agent vs the *strongest* baseline).
- Per-regime + overall tables and a figure show the agent vs the baseline panel.
- New unit tests pass; `scikit-learn` added to env + lock; artifacts persisted;
  authored Dhruvi, no co-author.
- Honest framing preserved (per-fold/seed JK primary; DSR cross-check). Expectation
  from §5: the agent underperforms the strong baselines net of cost — disclose it
  plainly, and note any regime where it is competitive.

### 8.6 Suggested commands (zsh-safe — no inline comments)

```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization
pip install scikit-learn pytest >/dev/null 2>&1
make lock
make test
caffeinate -i python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 --min-train-months 12 --episodes 200 --config tuning/best_config.json
caffeinate -i python experiments/multi_seed.py --seeds 0 1 2 3 4 --episodes 500 --config tuning/best_config.json
git add utils/baselines.py experiments/walk_forward_eval.py experiments/multi_seed.py main.py utils/plotting.py test/test_baselines.py environment.yml requirements.txt requirements.lock.txt experiments/results/walk_forward_*.json experiments/results/walk_forward_*.csv experiments/results/*.png
git commit -m "Phase 4: strong baseline overhaul (SPY B&H, 60/40 SPY/AGG, risk parity, rolling Ledoit-Wolf MVO)"
```

To avoid re-training the agent, you may instead reuse the cached per-fold agent
returns and only recompute baselines + significance — add a `--reuse-agent-returns`
path to the harness that loads the saved net return series if you build one.

### 8.7 Explicitly OUT of scope for Phase 4

- Entropy/reward fixes and the **turnover penalty / cost-aware reward** (I-6, I-10,
  the α blow-up) → **Phase 5**. (This is what would actually fix the net result.)
- Reproducibility nondeterminism hardening (§12) → **Phase 5**.
- README/doc rewrite, state-dim/FinRL fixes (I-9) → **Phase 6**.
- Re-tuning HPO with longer proxies (I-8 transfer gap) → **Phase 5/6**.

---

## 9. Roadmap beyond Phase 4 (context, not action)

- **Phase 5** — entropy/learning-signal + **the turnover problem** (the binding
  constraint from §5): explicitly fix the entropy temperature pathology (collapse ↔
  blow-up), entropy floor / scheduled α / re-derived target entropy; reward-scaling
  sweep; **turnover penalty / cost-aware reward**; ablate MLP vs transformer and
  ± FinBERT. Also fix the reproducibility nondeterminism (§12) and consider a
  longer-episode HPO proxy (I-8).
- **Phase 6** — sync docs to code (state-dim **144**, "FinRL-compatible" not
  -powered, repo tree), rewrite results around the leak-free CI-backed multi-regime
  numbers, add a Limitations section + one-command reproduction recipe.

Then a **Capstone Roadmap** (only after results are trustworthy): Docker, W&B +
Hydra tracking, GitHub Actions CI, a deployed Streamlit/Gradio demo, algorithm
breadth (PPO/TD3 + CVaR risk-aware reward), and a GenAI extension (LLM
news/filings → signals, RAG over 10-Ks, LLM rebalance rationales).

---

## 10. Definition of Done (whole remediation effort)

- ✅ All runs seeded, version-pinned, commit+config stamped (Phase 0).
- ✅ Every headline metric = mean ± 95% CI across ≥5 seeds (single-window) / ≥3
  seeds × folds (walk-forward), named significance test vs benchmark (Phases 1, 3).
- ✅ No leakage in HPO or universe; walk-forward leak-free by construction; all
  unit-test-guarded and documented (Phases 2, 3).
- ✅ Results across ≥3 market regimes (Phase 3 — 4 regimes, 27 folds).
- ⬜ **Strong baseline set (Phase 4 — NEXT).**
- ◑ Policy behavior explained (✅ Phase 1); entropy mechanism + turnover fixed or no
  longer claimed (Phase 5).
- ⬜ README matches code, has a Limitations section + one-command repro (Phase 6).

---

## 11. Gotchas / lessons carried forward

- **Always report NET (value-path) Sharpe.** `trainer.backtest()` returns **gross**
  metrics; `multi_seed._net_metrics()` and `walk_forward._fold_net_metrics()` wrap
  it. Phase 4 baselines already return net — keep it that way. This exact bug has
  bitten the project three times.
- **Per-fold / per-seed JK is primary**; pooled/bootstrap/DSR are cross-checks.
- **`caffeinate -i` every long run.** A Phase-3 fold stalled 2h25m from the Mac
  sleeping. The expanding training window makes later walk-forward folds much slower
  than earlier ones — budget accordingly.
- **zsh does not treat `#` as a comment.** Give comment-free command blocks.
- **Ray is broken in this env and you should not fight it.** The HPO auto-falls-back
  to a Ray-free random search with the same leak-free objective.
- **scikit-learn is not installed yet** — Phase 4 must add it (for Ledoit–Wolf) and
  `make lock`. Guard the import.
- **Reproducibility nondeterminism (open issue).** Two identical-seed runs give
  different per-seed results (seed 0 OOS Sharpe 0.17 vs 0.40); this also widens the
  walk-forward bands (2022-bear std 1.47). Likely nondeterministic CPU torch ops or
  an unseeded RNG path (env `reset()` / `action_space.sample()` / buffer). Try
  `torch.use_deterministic_algorithms(True)`, `OMP_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, and audit every RNG source. Deferred to Phase 5.
- **Entropy α can blow up, not just collapse** (α ≈ 6 by ep 500 on the leak-free
  config, inflating turnover) — Phase 5.
- **Turnover, not signal, is the binding constraint** (gross +1.15 → net −0.44).
  Phase 4 measures it against strong baselines; Phase 5 is where a turnover
  penalty / cost-aware reward could actually move the net result.
- **Don't `git add -A`** — it commits large binaries. `.gitignore` covers them; add
  source + small summaries explicitly. Remove a stale `.git/index.lock` if git
  refuses.
- `pytest` and `scikit-learn` are not in the runtime lock; install before use, then `make lock`.
- **State-dim is 144 now** (24 assets), not 180/210 — older docs are stale (I-9).
