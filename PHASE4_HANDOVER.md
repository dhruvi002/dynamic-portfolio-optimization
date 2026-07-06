# SAC Portfolio Optimization — Phase 4 Handover & Phase 5 Brief

**Paste this whole document into a fresh Claude chat to continue the project.**
It contains everything needed: project context, what Phases 0–4 changed and
why, the exact environment that works (including the dependency landmines),
the current honest leak-free multi-regime results against a strong baseline
panel, and a complete, actionable specification for Phase 5 (entropy fix +
turnover-penalty / cost-aware reward — the change that would actually move the
net result).

- **Repo:** https://github.com/dhruvi002/dynamic-portfolio-optimization
- **Local path (Mac):** `~/Documents/Projects/finrl/dynamic-portfolio-optimization`
- **Conda env:** `portfolio-rl` (Python 3.10, macOS arm64, CPU-only)
- **Phase 0–2:** June 2026 · **Phase 3:** July 2026 · **Phase 4:** July 2026
- **HEAD at handover:** see `git log -1` — most recent commits are the Phase 4
  baseline overhaul + the post-hoc fixes below (Series-truthiness crash in
  `multi_seed.py`, yfinance per-fold-download stall in `walk_forward_eval.py`)
- **Owner:** Dhruvi (MS-CS student targeting entry-level Data / ML /
  Applied-Research / GenAI roles)

---

## 0. Standing instructions for the assistant (read first)

1. **Honest reporting over big numbers.** Every headline metric ships with a
   confidence interval and a significance test. Never re-introduce a bare
   "+24%"/"+47%" headline. If an edge is not statistically significant, say so.
   Report **net-of-cost** metrics — `utils/trainer.backtest()` returns **gross**
   metrics; wrap them with the net value-path computation (this has bitten the
   project three times now; see §12).
2. **Reproducibility before results.** Runs are seeded, version-pinned, and
   stamped (`run_meta.json`). Keep it intact. There is a known residual
   nondeterminism on this Mac (§12) — **Phase 5 fixes this, don't defer again.**
3. **No leakage.** Walk-forward and every Phase-4 baseline estimator (risk
   parity, rolling Ledoit-Wolf MVO) are leak-free by construction — rolling
   estimators use only data with `date <= rebalance_date` / `date < test_start`.
   All guarded by unit tests (`test/test_no_leak.py`, `test/test_walk_forward.py`,
   `test/test_baselines.py`). **Any new work in Phase 5 must preserve this.**
4. **CPU-first.** Model is small (MLP, 24 assets, state-dim 144). Phase 3's
   walk-forward (9 folds × 3 seeds × 200 ep) took **~40h+** wall-clock on this
   MacBook Air in the Phase 4 re-run (expanding window; later folds are much
   slower). Phase 1's single-window (5 seeds × 500 ep) took **~40h** as well.
   Budget accordingly for Phase 5 re-runs — they retrain the agent, unlike
   Phase 4's baseline-only work.
5. **`caffeinate` every long run.** `caffeinate -i python experiments/....py …`.
   A stalled Mac has cost hours before.
6. **yfinance: download once, slice per-window, don't call per-fold.** A
   pre-fix Phase 4 walk-forward run called `spy_buy_and_hold`/`spy_agg_60_40`
   once per fold per seed (54 total calls) and one fold stalled **33 hours**
   on what looks like a Yahoo Finance rate limit. Fixed with
   `spy_buy_and_hold_from_series`/`spy_agg_60_40_from_series` + a module-level
   `_yf_download` cache in `utils/baselines.py`; `walk_forward_eval.py`
   downloads SPY/AGG **once** for the full span up front
   (`_load_spy_agg_full_span`) and slices per fold. **Any new per-window
   network call added in Phase 5 must follow this download-once pattern.**
7. **Git: do NOT add Claude / Sonnet / Opus as a co-author.** Commit regularly,
   authored as the user only. No `Co-authored-by` trailers. Do **not** `git add -A`
   blindly — it sweeps in large `.pt`/`.npz`/`.pkl` binaries; `.gitignore` covers
   `checkpoints/ runs/ plots/ experiments/results/checkpoints/ *.pt *.pkl *.npz`.
   Commit source + the small JSON/CSV/PNG summaries explicitly. If git refuses
   with a stale `.git/index.lock`, remove that file and retry.
8. **Their shell is zsh, which does NOT treat `#` as a comment by default** — put
   no inline `# comments` on command lines. Keep command blocks comment-free.
9. **Use the existing harnesses.** `experiments/multi_seed.py` (`make evaluate`)
   is the single-window instrument; `experiments/walk_forward_eval.py`
   (`make walkforward`) is the multi-regime instrument. Both now compare the
   agent against the **full Phase-4 baseline panel**, not just equal-weight.
   Reuse their aggregation/significance helpers, don't reinvent them.
10. **Never assert a python subprocess/agent's task is done without running the
    tests.** The Phase 4 code was verified with `pytest` in an isolated
    environment that lacked torch/gymnasium/ta before the user's own
    `make test`/full run confirmed it end-to-end; a `Series`-truthiness crash
    (`x.get(...) or y` on a pandas Series) still slipped through and only
    surfaced ~40 hours into a real run. Prefer `x is None` checks over
    truthiness on anything that could be a Series/array.

---

## 1. Project in one paragraph

A deep reinforcement-learning portfolio optimizer for a fixed large-cap US
equity universe. A **Soft Actor-Critic (SAC)** agent with a **Dirichlet policy
on the K-simplex** allocates weights across 24 continuous-membership DJIA
names. The environment is a hand-rolled Gymnasium env (FinRL-*compatible*, not
FinRL-powered). The codebase is strong ML engineering — twin critics, a
Welford running normalizer that freezes at eval, chronological train/val/test
split, checkpoint selection on validation Sharpe, transaction + slippage
costs applied identically to agent and baselines, an optional cross-asset
transformer encoder, and an optional FinBERT sentiment channel. There is a
leak-free HPO runner, a single-window honest-evaluation harness (Phase 1), a
multi-regime walk-forward harness (Phase 3), and now (Phase 4) both harnesses
compare the agent against a **strong, standard baseline panel** — SPY
buy-and-hold, the real 60/40 SPY/AGG, inverse-vol risk parity, and rolling
Ledoit-Wolf MVO — not just equal-weight. **The original headline ("+24%
Sharpe / +47.3% return over equal-weight") rested on a single unseeded, leaky
backtest.** Phases 0–4 replaced it with a reproducible, multi-seed,
CI-backed, significance-tested, leakage-free, multi-regime evaluation against
a benchmark panel a practitioner would respect. **The honest verdict, now
confirmed against the strongest baseline in every regime and every seed: net
of cost the agent does not beat any baseline anywhere (0/27 walk-forward
folds, 0/10 single-window baselines). Gross Sharpe (+1.26–1.39) is
competitive pre-cost — this is a turnover-cost problem, not a signal problem,
and the deficit vs the strongest baseline is larger than vs equal-weight
alone.**

---

## 2. Issue register (full remediation backlog)

Severity: **H** = blocks the headline claim, **M** = materially weakens credibility, **L** = polish.

| ID | Issue | Sev | Phase | Status |
|----|-------|-----|-------|--------|
| I-1 | No global seeding → not reproducible | H | 0 | ✅ DONE |
| I-2 | Single backtest/seed/regime; no significance/CI | H | 1, 3 | ✅ DONE |
| I-3 | HPO leaks the test set | H | 2 | ✅ DONE |
| I-4 | Survivorship / look-ahead universe (SHW back-filled) | H | 2 | ✅ DONE |
| I-5 | IS-vs-OOS gap unexplained; no turnover/concentration/active-share | H | 1 | ✅ DONE |
| I-6 | Entropy temperature pathology (collapse ↔ blow-up) | M | 5 | ◑ MEASURED (fix → Phase 5) |
| I-7 | Weak/strawman baselines; SPY buy-and-hold missing | M | 4 | ✅ **DONE** |
| I-8 | Config ambiguity; 15-ep HPO proxy may not transfer to 500-ep | M | 5, 6 | ◑ PARTIAL (leak-free; transfer gap remains) |
| I-9 | Doc/code mismatches (state-dim **144**, FinRL claim, tree) | L | 6 | ⬜ TODO |
| I-10 | Tiny reward signal (1e-4 scaling); learning-signal not validated | M | 5 | ⬜ **TODO — DO THIS NEXT (with I-6)** |
| I-11 | Headline overstates precision | M | 1 | ✅ DONE |

Dependencies: **0** → **1** → **2** (leak removal) → **3** (regimes) →
**4** (strong baselines — done) → **5** (entropy/reward/turnover — next) →
**6** (docs + honest claims).

---

## 3. What Phases 0–3 did (context — see PHASE0/1/2/3_HANDOVER.md + PHASE1/2/3_NOTES.md for full detail)

- **Phase 0** — determinism + self-documentation (`utils/seeding.py`,
  `utils/run_meta.py`, `environment.yml`, `requirements.lock.txt`, `Makefile`).
- **Phase 1** — statistically defensible single-window evaluation: multiple
  seeds, bootstrap CIs, Jobson–Korkie–Memmel Sharpe-difference test, PSR/DSR/PBO,
  policy diagnostics (turnover/HHI/active share). **Convention: per-seed JK is
  PRIMARY; pooled/bootstrap are optimistic cross-checks.**
- **Phase 2** — leakage-free: killed the HPO test-set leak and the
  survivorship/look-ahead universe (`config.UNIVERSE` = 24 continuous DJIA
  members 2018-01→2025-01; state-dim **144**).
- **Phase 3** — multi-regime walk-forward (`utils/walk_forward.py`,
  `experiments/walk_forward_eval.py`, `utils/regimes.py`): NET-of-cost metrics
  per fold, leak-free by construction (`train_end < test_start` asserted),
  per-regime + overall aggregation, per-fold JK vs equal-weight. **Phase-2
  single-window verdict generalizes: agent loses to EW in 0/27 folds across
  COVID/2021-bull/2022-bear/2023-24-recovery.**

---

## 4. What Phase 4 did (THIS WAS THE NEW WORK)

**Objective:** replace the weak baseline set with a strong, standard panel and
re-measure the agent against it in both harnesses. Resolves **I-7**. Full
design notes + complete result tables: `PHASE4_NOTES.md`.

### 4.1 New baselines (`utils/baselines.py`)
`spy_buy_and_hold` (single entry cost, held), `spy_agg_60_40` (real 60/40,
kept `spy_qqq` for backward-compat), `risk_parity` (inverse-vol, rolling
252-day, no look-ahead), `rolling_mvo_ledoit_wolf` (min-var + max-Sharpe,
Ledoit-Wolf shrinkage re-estimated every rebalance, sample-covariance fallback
if `scikit-learn` missing). Same cost model, same `(metrics, values, dates)`
signature as existing baselines. `scikit-learn` added to
`requirements.lock.txt`/`requirements.txt`/`environment.yml`.

### 4.2 Wired into both harnesses
`main._collect_baselines_full` runs the full 10-strategy panel.
`experiments/multi_seed.py` runs per-seed JK vs **every** baseline and reports
vs the **strongest** as the headline (kept EW comparison too).
`experiments/walk_forward_eval.py` computes every baseline per fold
(estimators leak-free: `train_df_fold = df[date < test_start]`), extends
`walk_forward_per_fold.csv` with per-baseline columns, and reports agent vs
strongest-baseline-per-regime as the Phase-4 headline table.

### 4.3 Figure
`utils.plotting.plot_walk_forward_baseline_panel` — full panel, grouped bars,
per regime, 95% CI. → `experiments/results/walk_forward_baseline_panel.png`.

### 4.4 Tests
`test/test_baselines.py` extended to 40 tests (30 new): cost-model consistency,
no-look-ahead (prefix-invariance under truncated test windows), Ledoit-Wolf
shrinkage + no-sklearn fallback, SPY B&H one-time entry cost. All mocked
(no live yfinance calls in CI).

### 4.5 Post-hoc bug fixes (found by the user's real run, not by tests)
1. `experiments/multi_seed.py`: `baseline_series.get("Equal Weight") or
   _equal_weight_return_series(test_df)` raised `ValueError: The truth value
   of a Series is ambiguous` — crashed **after** all 5 seeds had finished
   training (~40h). Fixed to an explicit `is None` check. **Lesson: never use
   `or`/truthiness on anything that could be a pandas Series/ndarray — this
   passed local unit tests because they never exercised this exact code path
   with a real baseline_series dict.**
2. `experiments/walk_forward_eval.py` + `utils/baselines.py`: per-fold
   `spy_buy_and_hold`/`spy_agg_60_40` calls (54 total across 3 seeds × 9 folds)
   caused one fold to stall 33 hours (likely a Yahoo Finance rate limit).
   Fixed by downloading SPY/AGG once for the full span
   (`_load_spy_agg_full_span`) and slicing per fold
   (`spy_buy_and_hold_from_series`/`spy_agg_60_40_from_series`).

### 4.6 Files changed in Phase 4
- **`utils/baselines.py`** — 4 new baseline fns + `_from_series` variants +
  `_yf_download` cache + `_shrunk_covariance` + `_inverse_vol_weights`.
- **`main.py`** — `_collect_baselines_full`; `_collect_baselines` now a wrapper.
- **`experiments/multi_seed.py`** — `_baseline_return_series`,
  `_collect_baseline_return_series`, `run_significance` now loops over every
  baseline, `_jk_vs_one_baseline` extracted.
- **`experiments/walk_forward_eval.py`** — `BASELINE_SPECS`,
  `_baseline_fold_returns`, `_load_spy_agg_full_span`, extended
  `_summarise_rows`/`_print_report`/persisted JSON with `_counts_vs_best`.
- **`utils/plotting.py`** — `plot_walk_forward_baseline_panel`.
- **`test/test_baselines.py`** — 30 new tests.
- **`requirements.lock.txt`, `requirements.txt`, `environment.yml`** —
  scikit-learn added.
- **`PHASE4_NOTES.md`, `PHASE4_HANDOVER.md`** — NEW (this document + companion).

---

## 5. Current honest result (Phase 4: strong baseline panel)

Full detail + all 10 baselines + all per-seed/per-fold JK tests in
`PHASE4_NOTES.md`. Headline:

### 5.1 Single-window (5 seeds × 500 ep, test 2023-01 → 2025-01)
Agent NET Sharpe **+0.082 ± 0.654** [−0.468, +0.631]; gross **+1.387 ± 0.270**
(cost drag **+1.305** at mean turnover 0.286/step). Strongest baseline is
**SPY/QQQ 60/40** (Sharpe +1.980). Agent loses to **all 10 baselines**;
significant (5% JK) against 8/10, including the strongest (4/5 seeds, median
p=4.3e-4, ΔSharpe(annual)=−1.838±0.646). DSR = 0.0193 (up from ≈0.004–0.0045 in
Phases 1–3, still ≈0).

### 5.2 Walk-forward (3 seeds × 9 folds × 200 ep, full 2019-04→2025-01 span)
Overall NET Sharpe **−0.368 ± 1.742** (gross +1.260). Beats EW in 0/27 folds;
beats the **strongest baseline per fold** in **0/27** folds too, with a
*larger* deficit (ΔSharpe(annual) vs strongest = **−2.079±0.847** vs −1.709±0.694
vs EW alone). 18/27 folds JK-significant vs the strongest baseline. DSR=0.0010.

### 5.3 Interpretation
The tougher benchmark panel **sharpens, not softens**, Phases 2–3's
conclusion. Gross Sharpe is genuinely competitive with the strong baselines
pre-cost (this is not a signal-quality problem). The transaction-cost drag at
~0.29–0.4/step turnover is what erases every bit of edge. **Turnover, not
signal, is the binding constraint — Phase 5 is where a cost-aware reward /
turnover penalty could actually move the net result; nothing before it has
touched the mechanism.**

---

## 6. The environment that works (authoritative)

Conda env **`portfolio-rl`**, Python **3.10**, macOS arm64, CPU. Top-level pins
(update via `make lock` after any install):

```
torch==2.2.2        numpy==1.26.4      pandas==2.2.2       scipy==1.13.1
gymnasium==0.29.1   yfinance==1.5.1    ray[tune]==2.9.3    hyperopt==0.2.7
matplotlib==3.8.4   seaborn==0.13.2    tensorboard==2.16.2 tqdm==4.66.4
ta==0.11.0          transformers==4.40.2   scikit-learn==<run make lock to capture>
```

**scikit-learn is now required** (Ledoit-Wolf shrinkage in `utils/baselines.
rolling_mvo_ledoit_wolf`). Guarded lazy-import with sample-covariance fallback
if absent, so the module still imports either way.

**Dependency landmines:**
- **`pytest`** is a test-only dep, not in the runtime lock — `pip install
  pytest` before `make test`, then `make lock`.
- **Ray 2.9.3 is broken** against modern setuptools/pyarrow. HPO auto-falls
  back to a Ray-free random search with the identical leak-free objective.
- `finrl`, `pyfolio-reloaded` intentionally **excluded**.
- **yfinance rate limits under rapid repeated calls** — see §0.6. Don't call
  `spy_buy_and_hold`/`spy_agg_60_40` in a per-fold/per-window loop; download
  once and slice, or use the `*_from_series` variants.

`make test` should run **136+ tests** (106 pre-Phase-4 + 30 new baseline
tests). After any clean install run `make lock`.

---

## 7. Repo map (key files)

```
main.py                      # CLI: train / tune / backtest / walkforward (+ --seed)
                             #   _collect_baselines_full / _collect_baselines (Phase 4)
config.py                    # date windows + UNIVERSE (24) + EXCLUDED_FROM_DJ30
agent/sac.py                 # SACAgent, DirichletActor, Critic, ReplayBuffer,
                             #   AssetTransformerEncoder; update() logs policy_entropy
                             #   ← PHASE 5 TOUCHES THIS (entropy fix)
environment/portfolio_env.py # Gym env; UNIVERSE class attr (alias DJ30_TICKERS);
                             #   state-dim 144; info.port_return is GROSS of cost
                             #   ← PHASE 5 TOUCHES THIS (reward shaping / turnover penalty)
utils/trainer.py             # train(...seed=), backtest(...) — backtest is GROSS
utils/metrics.py             # compute_sharpe/sortino/calmar/max_drawdown/all_metrics
utils/baselines.py           # equal_weight, spy_qqq, momentum_12_1, min_variance,
                             #   max_sharpe_mvo (static, legacy) PLUS Phase 4:
                             #   spy_buy_and_hold, spy_agg_60_40, risk_parity,
                             #   rolling_mvo_ledoit_wolf (+ _from_series variants,
                             #   _yf_download cache, _shrunk_covariance)
utils/walk_forward.py        # NET metrics, _fold_windows, _fold_net_metrics,
                             #   per-fold diagnostics + leak assertion
utils/regimes.py             # regime bands + regime_for + regimes_present
utils/normalizer.py          # RunningNormalizer (Welford; train()/eval(); skips weights)
utils/significance.py        # JK–Memmel, bootstrap CI, PSR/DSR, PBO
utils/diagnostics.py         # turnover/HHI/active share, returns_from_values,
                             #   aggregate_metrics, bootstrap_ci
utils/plotting.py            # + plot_walk_forward_baseline_panel (Phase 4)
experiments/multi_seed.py    # single-window harness (make evaluate); Phase 4:
                             #   per-seed JK vs every baseline, strongest_baseline
experiments/walk_forward_eval.py # multi-regime harness (make walkforward); Phase 4:
                             #   BASELINE_SPECS, per-fold full-panel JK,
                             #   _load_spy_agg_full_span (download-once fix)
experiments/results/         # per_seed_results.csv, aggregate_metrics.json,
                             #   significance.json, baselines.json,
                             #   walk_forward_per_fold.csv, walk_forward_regime.json,
                             #   walk_forward_significance.json,
                             #   walk_forward_regimes.png, walk_forward_baseline_panel.png
tuning/tune_runner.py        # LEAK-FREE HPO; Ray Tune w/ Ray-free fallback
tuning/best_config.json      # leak-free tuned HPs
data/pipeline.py             # download/indicators/splits; DJ30_TICKERS ← UNIVERSE
data/sentiment_pipeline.py   # FinBERT sentiment (optional; would raise state-dim)
test/                        # pytest suite incl. test_no_leak.py, test_walk_forward.py,
                             #   test_baselines.py (40 tests, 30 new in Phase 4)
Makefile, environment.yml, requirements.lock.txt, requirements.txt
PHASE0/1/2/3_HANDOVER.md, PHASE1/2/3_NOTES.md  # prior phase records
PHASE4_NOTES.md               # Phase 4 design notes + full results (companion to this doc)
```

Useful signatures added/changed in Phase 4:
```python
from utils.baselines import (
    equal_weight, spy_qqq, momentum_12_1, min_variance, max_sharpe_mvo,   # existing
    spy_buy_and_hold, spy_agg_60_40, risk_parity, rolling_mvo_ledoit_wolf,  # Phase 4
    spy_buy_and_hold_from_series, spy_agg_60_40_from_series,               # download-once variants
)
from main import _collect_baselines_full   # {name: (metrics, values, dates)}
```

---

## 8. PHASE 5 — Entropy fix + cost-aware reward / turnover penalty (DO THIS NEXT)

**Objective:** this is the phase that can actually change the net result.
Phases 1–4 have exhaustively *measured* that (a) the entropy temperature α
pathologically collapses or blows up, and (b) turnover cost is the entire gap
between gross (+1.26 to +1.39 Sharpe, competitive) and net (−0.37 to +0.08,
losing to every baseline). Nothing so far has touched the *mechanism*. Resolves
**I-6, I-10**; makes real progress on **I-8** (transfer gap) and §12
(reproducibility nondeterminism).

### 8.1 Task A — fix the entropy temperature pathology (I-6)
Evidence from Phase 4's own run: seed 0's α went from ~1.76 (fold-1-scale
short runs) to **9.58 by episode 500** in the single-window 500-episode run
(monotonically climbing every logged checkpoint: 2.74→5.79→7.94→8.91→9.58).
This is the "blow-up" side of the pathology documented in Phase 1 (the
"collapse" side was seen in some walk-forward folds with α→0.001-0.02). Both
directions are visible in real logs now, not just suspected.
- Investigate `agent/sac.py`'s automatic entropy tuning (the log-alpha update
  and target entropy derivation). Candidates: entropy floor/ceiling clamp,
  a scheduled (not fully automatic) α, re-derived target entropy for a
  Dirichlet policy on a 24-dim simplex (the standard `-action_dim` target
  entropy heuristic is derived for diagonal-Gaussian policies, not Dirichlet —
  it may simply be the wrong target for this action distribution).
- Log α and policy entropy every episode (`utils/plotting.plot_alpha_entropy`
  already exists — use it per seed, not just the median-seed representative).
- Acceptance: α trajectory does not exceed a documented bound (e.g. stays
  within [0.01, 5] or whatever the derived floor/ceiling is) across all seeds
  and folds tested.

### 8.2 Task B — turnover penalty / cost-aware reward (the big one)
- Add a turnover penalty term to the reward in `environment/portfolio_env.py`
  (currently `info.port_return` is GROSS of cost; the cost is applied to
  `portfolio_value` but not fed back into the reward the agent optimizes for
  cheaply/every step — confirm this by reading the reward computation before
  changing it). A cost-aware reward should let the agent *learn* to trade less
  when the edge doesn't cover the cost, rather than being penalized only via
  the backtest's already-realized-but-not-optimized-for cost.
- Sweep the discount and penalty scaling with a proper reward-scaling sweep
  (I-10: current reward scaling is ~1e-4, flagged as possibly too small a
  learning signal even before adding a turnover term).
- **This must stay leak-free** and use the same net-of-cost measurement
  convention (§0.1) — the fix targets net Sharpe directly since the training
  signal now reflects the cost the eval already measures.
- Re-run both harnesses (`multi_seed.py`, `walk_forward_eval.py`) after the
  change and compare against the Phase 4 baseline panel — the acceptance bar
  is not "improves Sharpe" but "closes some of the gross→net gap without
  breaking leak-freedom or reproducibility."

### 8.3 Task C — reproducibility nondeterminism (§12, deferred twice already)
Two identical-seed runs give different per-seed results (Phase 3: seed 0 OOS
Sharpe 0.17 vs 0.40 across runs; this widens walk-forward bands — 2022-bear
std was 1.47 in Phase 3, 1.57 in Phase 4). Try, in order: `torch.
use_deterministic_algorithms(True)`, `OMP_NUM_THREADS=1`, `torch.
set_num_threads(1)`, and audit every RNG source (env `reset()`,
`action_space.sample()`, replay buffer sampling, any `np.random` call not
routed through the seeded global RNG). Acceptance: `make repro` (already
exists as a 2-run diff target) reports identical logs for identical seeds.

### 8.4 Task D — ablations (lower priority, only after A–C)
MLP vs the existing transformer encoder; ± FinBERT sentiment. Only run these
once the entropy/turnover fixes are in, since a broken learning signal makes
any encoder/sentiment ablation uninterpretable.

### 8.5 Task E — longer-episode HPO proxy (I-8 transfer gap)
The current HPO objective trains for a short proxy (historically 15 episodes)
and the transfer to 200–500-episode real runs is untested. Consider a longer
proxy (e.g. 50–100 episodes) now that entropy/reward are being touched anyway
— re-tuning HPs after a reward-shape change is necessary regardless, so this
is a natural place to also address the proxy-length question.

### 8.6 Acceptance criteria (Phase 5)
- Documented, bounded α trajectory across all seeds/folds (no collapse-to-~0
  or blow-up-to-double-digits).
- A cost-aware reward term is implemented, leak-free, and measured against the
  Phase 4 baseline panel using the existing harnesses (no new ad-hoc
  evaluation code).
- `make repro` passes (identical seeds → identical logs).
- Honest framing preserved throughout (per-fold/seed JK primary; DSR
  cross-check; net-of-cost always, per §0.1). Report the new gross→net gap and
  whether it closed, disclosing plainly if it didn't.
- New unit tests for the reward/entropy changes; full suite still green.
- Authored Dhruvi, no co-author trailer.

### 8.7 Suggested commands (zsh-safe — no inline comments)
```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization
make test
caffeinate -i python experiments/multi_seed.py --seeds 0 1 2 3 4 --episodes 500 --config tuning/best_config.json
caffeinate -i python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 --min-train-months 12 --episodes 200 --config tuning/best_config.json
make repro
```

### 8.8 Explicitly OUT of scope for Phase 5
README/doc rewrite, state-dim/FinRL claim fixes, repo-tree sync (I-9) → Phase 6.

---

## 9. Roadmap beyond Phase 5 (context, not action)

- **Phase 6** — sync docs to code (state-dim **144**, "FinRL-compatible" not
  -powered, repo tree), rewrite results around the leak-free CI-backed
  strong-baseline numbers, add a Limitations section + one-command
  reproduction recipe.

Then a **Capstone Roadmap** (only after results are trustworthy): Docker,
W&B + Hydra tracking, GitHub Actions CI, a deployed Streamlit/Gradio demo,
algorithm breadth (PPO/TD3 + CVaR risk-aware reward), and a GenAI extension
(LLM news/filings → signals, RAG over 10-Ks, LLM rebalance rationales).

---

## 10. Definition of Done (whole remediation effort)

- ✅ All runs seeded, version-pinned, commit+config stamped (Phase 0).
- ✅ Every headline metric = mean ± 95% CI across ≥5 seeds (single-window) / ≥3
  seeds × folds (walk-forward), named significance test vs benchmark (Phases 1, 3).
- ✅ No leakage in HPO, universe, or any baseline estimator; all
  unit-test-guarded and documented (Phases 2, 3, 4).
- ✅ Results across ≥3 market regimes (Phase 3 — 4 regimes, 27 folds).
- ✅ **Strong baseline set, both harnesses (Phase 4).**
- ◑ Policy behavior explained (✅ Phase 1); entropy mechanism + turnover fixed
  or no longer claimed (Phase 5 — **NEXT**).
- ⬜ Reproducibility nondeterminism resolved (Phase 5).
- ⬜ README matches code, has a Limitations section + one-command repro (Phase 6).

---

## 11. Gotchas / lessons carried forward

- **Always report NET (value-path) Sharpe.** `trainer.backtest()` returns
  **gross** metrics; wrap with `_net_metrics`/`_fold_net_metrics`. This exact
  bug has bitten the project three times.
- **Per-fold / per-seed JK is primary**; pooled/bootstrap/DSR are cross-checks.
- **Headline = agent vs the STRONGEST baseline** as of Phase 4, not just
  equal-weight — report both, but lead with strongest.
- **`caffeinate -i` every long run.** Multi-hour trainings have stalled from
  Mac sleep before.
- **yfinance rate-limits under rapid repeated calls in a loop** (§0.6) — always
  download once for the full span and slice per window; never call a
  network-backed baseline inside a per-fold/per-window loop.
- **Never use truthiness (`or`, `if x:`) on anything that might be a pandas
  Series/ndarray** — use explicit `is None` checks. This exact bug crashed a
  ~40-hour run at the very last step in Phase 4.
- **zsh does not treat `#` as a comment.** Give comment-free command blocks.
- **Ray is broken in this env and you should not fight it.** Auto-falls-back
  to a Ray-free random search with the same leak-free objective.
- **scikit-learn is now required** (Ledoit-Wolf); guarded import, sample-cov
  fallback if absent.
- **Reproducibility nondeterminism is still open** (two identical-seed runs
  give different results; widens walk-forward CI bands) — **Phase 5, task C,
  don't defer a third time.**
- **Entropy α can blow up, not just collapse** — Phase 4's real run shows α
  reaching 9.58 by episode 500 on the leak-free config. Phase 5, task A.
- **Turnover, not signal, is the binding constraint** (gross ~+1.3 → net ~0 to
  −0.4, against every baseline in the panel, in every regime). Phase 5 task B
  is the only lever that can plausibly move this.
- **Don't `git add -A`** — commits large binaries. Add source + small
  summaries explicitly. Remove a stale `.git/index.lock` if git refuses.
- `pytest` and `scikit-learn` are not in the runtime lock by default; install
  before use, then `make lock`.
- **State-dim is 144** (24 assets), not 180/210 — older docs are stale (I-9).
