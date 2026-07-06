# Phase 5 — Entropy Fix + Cost-Aware Reward + Reproducibility (implementation notes)

Companion to `PHASE5_PLAN.md`. Records the code that was actually written for
Tasks A–C, one correction to the Phase 4 handover's premise, and — honestly —
exactly what has been validated in-sandbox vs. what still requires the
`portfolio-rl` conda env and the ~40h retrains before any result claim.

**Honest-reporting note (per handover §0.1):** nothing here claims the net
result improved. That claim can only be made after the retrains below. What is
done is the *mechanism* (entropy bound + attainable target, learnable turnover
penalty, thread/RNG determinism) plus unit tests; the acceptance bars that need
training runs are called out as OPEN.

---

## Correction to the Phase 4 handover premise (Task B)

The handover (§8.2) said the turnover cost "is applied to `portfolio_value` but
not fed back into the reward." Reading `environment/portfolio_env.py:step()`,
that is **not accurate**: the reward is `log(net / prev) · reward_scaling` with
`net = gross − tc − slip`, so it is *already net of cost*. The real problem is
**signal magnitude** (I-10): at ~0.29 turnover/step the realised cost is only a
few bps, buried under ~1% daily moves and then shrunk by `reward_scaling=1e-4`,
so the agent barely feels it. Task B was therefore implemented as an *explicit,
separately-weighted* turnover penalty, not as "adding cost that wasn't there."

---

## Task A — entropy target + bounded temperature (I-6)  `agent/sac.py`

- **New `dirichlet_symmetric_entropy(action_dim, conc)`** — differential
  entropy of a symmetric Dirichlet(conc·1_K) via the digamma closed form. Used
  to derive an *attainable* target. The Dirichlet entropy is maximised at the
  uniform policy (conc=1, `−lgamma(K)`; −51.6 for K=24) and decreases on both
  sides.
- **Root cause fixed:** the old `target_entropy = −(lgamma(K) + 0.5·K)` = −63.6
  (K=24) sat ~conc 5.5, *below* the uniform maximum, so the auto-tuner chased a
  concentration the clamped actor rarely reaches; with no bound `log_alpha` ran
  away (α→9.58 in the Phase 4 log) or collapsed (α→1e-3 in some folds). New
  default target = entropy of a **mild** symmetric Dirichlet (`target_conc=2.0`
  → −54.3, just below uniform) — an attainable fixed point that keeps the policy
  near-diversified, which also curbs turnover.
- **Bounded temperature:** `log_alpha` is clamped in-place after every update to
  `[log(alpha_min), log(alpha_max)]` (defaults α ∈ [0.01, 5.0]); `alpha_init`
  (default 1.0) is clamped into the band at construction.
- **All configurable:** `target_entropy` (explicit override), `target_conc`,
  `alpha_init`, `alpha_min`, `alpha_max` are SACAgent kwargs, threaded from
  `config` in `main.build_agent` and the walk-forward `agent_factory` with safe
  defaults (so an old `best_config.json` still works).

## Task B — cost-aware turnover penalty (I-10)  `environment/portfolio_env.py`

- **New env param `turnover_penalty` (λ_turnover, default 0.0)**; reward is now
  `(log_return − λ_turnover · Σ|Δw|) · reward_scaling`. Default 0.0 reproduces
  the pre-Phase-5 reward exactly.
- **Threaded through** `main.build_env` (training env only; eval metrics come
  from the value path, so the penalty is irrelevant there) and the walk-forward
  `env_kwargs`. Sweepable via `config["turnover_penalty"]` **or** a new
  `--turnover-penalty` CLI flag on both `experiments/multi_seed.py` and
  `experiments/walk_forward_eval.py` (flag overrides config).

## Task C — reproducibility / determinism (§12)  `utils/seeding.py`, `utils/trainer.py`

- **Thread pinning:** `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB` thread env vars set to 1
  at `utils.seeding` import time (before the backends spin up their pools), plus
  `torch.set_num_threads(1)` and best-effort `set_num_interop_threads(1)` inside
  `set_global_seed` as a runtime belt-and-braces. This removes the CPU
  float-reduction-order nondeterminism that was the likely §12 cause.
- **RNG-audit fix (primary leak):** `trainer.train()` now calls
  `env.action_space.seed(seed)` after reset. Gymnasium seeds `env.np_random`
  from `reset(seed=)` but `action_space.sample()` draws from a *separate*,
  otherwise-random RNG — so the 1000-step warm-up (and thus the whole replay
  buffer) previously differed across identical-seed runs.

---

## Files changed

- `agent/sac.py` — `dirichlet_symmetric_entropy`; new SACAgent entropy/α kwargs;
  attainable target derivation; `log_alpha` clamp in `update()`.
- `environment/portfolio_env.py` — `turnover_penalty` param + reward term + docs.
- `utils/seeding.py` — thread env vars + `set_num_threads(1)`.
- `utils/trainer.py` — `env.action_space.seed(seed)` in warm-up.
- `main.py` — `build_env(turnover_penalty=)`, `build_agent` entropy kwargs, train
  env wiring.
- `experiments/multi_seed.py` — `--turnover-penalty` + config wiring.
- `experiments/walk_forward_eval.py` — agent-factory entropy kwargs, env_kwargs
  turnover penalty, `--turnover-penalty`.
- `test/test_env.py` — 4 turnover-penalty tests.
- `test/test_sac.py` — 6 entropy-derivation / α-clamp tests.
- `test/test_seeding.py` — NEW: thread-pin + action-space-seeding determinism.
- `PHASE5_PLAN.md`, `PHASE5_NOTES.md` — plan + these notes.

## Validation status

**Verified in-sandbox (no torch needed):**
- `test/test_env.py` 16/16 pass (incl. new turnover-penalty tests).
- `test/test_seeding.py` pass (torch thread-count test skipped without torch).
- Combined torch-free suite: 49 passed, 1 skipped. All edited files byte-compile.
- Entropy derivation cross-checked against scipy (K=24, conc=2 → −54.292).

**OPEN — needs `portfolio-rl` conda env (has torch):**
- `make test` full suite (the torch-dependent `test/test_sac.py` additions —
  α-clamp and target-entropy tests — and `test/test_all.py`).
- `make repro` must report identical logs for identical seeds (Task C
  acceptance).

**OPEN — needs retrains (~40h each, `caffeinate -i`):**
- Confirm the α trajectory stays inside [0.01, 5] across all seeds/folds
  (Task A acceptance) via `plot_alpha_entropy` per seed.
- Sweep `--turnover-penalty` and re-run both harnesses vs the Phase 4 panel;
  report whether the gross→net gap closed (Task B acceptance) — honestly,
  including if it did not.
- Re-tune HPs after the reward-shape change (I-8 / Task E), ideally with a
  longer proxy than 15 episodes.

## Run recipe (zsh-safe, no inline comments)

```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization
make test
make repro
caffeinate -i python experiments/multi_seed.py --seeds 0 1 2 3 4 --episodes 500 --config tuning/best_config.json --turnover-penalty 0.5
caffeinate -i python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 --min-train-months 12 --episodes 200 --config tuning/best_config.json --turnover-penalty 0.5
```
