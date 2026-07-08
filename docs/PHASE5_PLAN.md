# Phase 5 — Entropy Fix + Cost-Aware Reward + Reproducibility (implementation plan)

Companion to `PHASE4_HANDOVER.md` §8. Resolves **I-6** (entropy pathology),
**I-10** (reward-signal magnitude), closes §12 (reproducibility
nondeterminism), and makes partial progress on **I-8** (HPO transfer gap).

This is a *plan*, not code. It records the exact mechanism behind each Phase 5
task as verified against the current `HEAD` (`c796a86`), the concrete fix
options with their file/line touch-points, acceptance bars, and the run
recipe. Written before any code change so the diagnosis is on record.

Honesty conventions from `PHASE4_HANDOVER.md` §0 hold throughout: net-of-cost
metrics, per-seed / per-(seed,fold) JK primary, DSR cross-check, leak-free,
no `Co-authored-by` trailer, zsh-safe command blocks (no inline `#`).

---

## 0. Findings from reading the code (verified, not assumed)

### 0.1 Entropy — the α blow-up is a mis-calibrated target, not a bug elsewhere
`agent/sac.py:232`:

```python
self.target_entropy = -(math.lgamma(action_dim) + action_dim * 0.5)
```

For the leak-free universe **K = action_dim = 24** this evaluates to
**−63.61 nats**, while the uniform Dirichlet (all α_i = 1) has differential
entropy **−lgamma(24) = −51.61 nats**. The target therefore demands entropy
*below* the uniform policy's — i.e. it asks the policy to be more concentrated
than uniform.

The α update (`agent/sac.py:293`):

```python
alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
```

drives `E[-log_pi] → target_entropy = -63.61`, equivalently `E[log_pi] → 63.61`.
The `DirichletActor` clamps concentrations to `CONC_MAX = 50` (`sac.py:97`), so
the sampled `log_pi` rarely reaches ~63.6. While `log_pi < 63.6` the bracket is
negative on average and **`log_alpha` climbs every step** — matching the Phase 4
single-window log (α: 2.74 → 5.79 → 7.94 → 8.91 → **9.58** by ep 500). As α
grows, the entropy bonus in the actor loss (`sac.py:285`) pushes the actor back
toward uniform, moving `log_pi` *further* from 63.6 → positive-feedback runaway.

The collapse direction seen in some walk-forward folds (α → 0.001–0.02) is the
same loop in reverse: when the critic gradient concentrates the policy hard
enough to overshoot the target (`log_pi > 63.6`), the bracket flips positive and
α decays toward 0. **One mis-calibrated target + no clamp explains both
directions.** The `+ 0.5·action_dim` term (12 nats at K=24) is what pushes the
target below the uniform entropy; that term is the proximate cause.

### 0.2 Reward — it is ALREADY net of cost; the problem is signal magnitude
`PHASE4_HANDOVER.md` §8.2 says the cost "is applied to `portfolio_value` but not
fed back into the reward the agent optimizes for," and asks to confirm by
reading the code. **Confirmed — and the claim needs correcting.**
`environment/portfolio_env.py:179-185`:

```python
gross = self.portfolio_value * port_return
net   = gross - tc - slip
self.portfolio_value = max(net, 1.0)
log_return = np.log(max(net, 1e-8) / prev_value)
reward = float(log_return * self.reward_scaling)
```

The reward is `log(net / prev) · reward_scaling`, and `net` already subtracts
`tc + slip`. So the reward **is** net of turnover cost. What is *gross* is only
`info["port_return"]` (the diagnostic field), not the reward.

The real issue is **magnitude / signal-to-noise**, which is I-10:
- At mean turnover ≈ 0.29/step the cost is `(tc_rate + slip_rate)·turnover`
  = `0.002 · 0.29` ≈ **5.8 bps/step**, buried under ~1%+ daily market moves in
  the same `log_return`.
- `reward_scaling = 1e-4` (`portfolio_env.py:47`) then shrinks the whole reward,
  so the gradient the cost contributes is negligible.

**Consequence for Task B:** the fix is not "add cost to a cost-free reward"; it
is "add an *explicit, separately-weighted* turnover penalty so the cost becomes
a learnable signal, and revisit `reward_scaling`." This reframes I-10 and B
together.

### 0.3 Reproducibility — CPU thread nondeterminism is uncontrolled
`utils/seeding.py` seeds python / numpy / torch / cuDNN and calls
`torch.use_deterministic_algorithms(True, warn_only=True)`, but:
- never sets `torch.set_num_threads(1)` or `OMP_NUM_THREADS=1`; and
- `warn_only=True` means nondeterministic ops warn rather than error, so they
  can still run nondeterministically.

On this CPU-only Mac, multi-threaded intra-op float reduction order is the most
likely source of the two-runs-differ symptom (§12). This is the cheapest task
to validate — no retrain needed, just `make repro`.

---

## 1. Task A — fix the entropy temperature pathology (I-6)

**Touch-points:** `agent/sac.py` (`SACAgent.__init__` target-entropy line 232,
`log_alpha` init 233, `update()` α block 292-298); `utils/plotting.py`
(`plot_alpha_entropy`, per-seed).

**Changes:**
1. **Re-derive `target_entropy`** so it is *attainable* by the clamped Dirichlet
   actor. The standard `-action_dim` heuristic is for diagonal-Gaussian
   policies and is wrong here; the current `-(lgamma+0.5K)` over-corrects below
   the uniform entropy. Decision between the two approaches below is deferred to
   implementation (prototype both, keep whichever gives a bounded, stable α on
   short runs):
   - *Empirical calibration:* sweep Dirichlet concentrations across
     `[CONC_MIN, CONC_MAX]`, measure the attainable entropy range, set the
     target to a fraction inside it (e.g. a fixed margin below the uniform
     −51.61 rather than below it).
   - *Principled default + clamp:* set target to a small margin **below**
     uniform-Dirichlet entropy and lean on the α clamp for stability.
2. **Clamp `log_alpha`** to a documented band each update (e.g. α ∈ [0.01, 5];
   exact bounds recorded once derived). This alone breaks the runaway even if
   the target is imperfect.
3. **Per-seed α + entropy logging** every episode via `plot_alpha_entropy`
   (not just the median-seed representative), so the acceptance check covers all
   seeds/folds.

**Acceptance:** α trajectory stays within the documented bound across every seed
and fold tested — no collapse-to-~0, no double-digit blow-up.

## 2. Task B — cost-aware reward / turnover penalty (I-10 + the net-result lever)

**Touch-points:** `environment/portfolio_env.py` (`step()` reward, lines
163-185; `reward_scaling` default line 47, plus wherever it is threaded from
config/CLI).

**Changes:**
1. Add an **explicit turnover-penalty term** to the per-step reward, separate
   from the value-path cost already present:
   `reward = (log_return - lambda_turnover * turnover) * reward_scaling`
   (with `turnover = Σ|Δw|`, already computed at `portfolio_env.py:165`). Keeping
   it separate makes cost sensitivity a single tunable knob rather than a
   quantity diluted inside a noisy log-return.
2. **Sweep `lambda_turnover`** and revisit `reward_scaling` (I-10 — current 1e-4
   may be too small a learning signal even before the penalty). A proper
   reward-scaling sweep, discount included.
3. Stay **leak-free** and keep the net-of-cost measurement convention (§0.1 of
   the handover). Re-run both harnesses unchanged and compare to the Phase 4
   panel.

**Acceptance:** *not* "improves Sharpe" but "closes some of the gross→net gap
(currently +1.26–1.39 gross vs −0.37–+0.08 net) without breaking leak-freedom
or reproducibility." Report the new gap and disclose plainly if it did not
close. New unit tests for the reward term; full suite green.

## 3. Task C — reproducibility nondeterminism (§12, do not defer a third time)

**Touch-points:** `utils/seeding.py`; process entry points that import torch
(`experiments/multi_seed.py`, `experiments/walk_forward_eval.py`, `main.py`).

**Changes (in order, stop when `make repro` passes):**
1. `torch.set_num_threads(1)` inside `set_global_seed`, and export
   `OMP_NUM_THREADS=1` **before torch is imported** (set in the entry-point
   process env, not just after import).
2. Consider `use_deterministic_algorithms(True)` without `warn_only` on CPU to
   surface any remaining nondeterministic op.
3. **Audit every RNG source:** env `reset()` (`portfolio_env.py` uses both the
   gym base RNG via `super().reset(seed=)` and `self._rng = default_rng(seed)`),
   `env.action_space.sample()` during warm-up (`trainer.py:55`), the warm-up
   `env.reset()` on `done` that passes no seed (`trainer.py:59`), and
   `ReplayBuffer.sample` (`random.sample`, `sac.py:32`).

**Acceptance:** `make repro` (existing 2-run diff target, `Makefile:50`) reports
identical logs for identical seeds.

## 4. Tasks D & E (only after A–C land)

- **D — ablations:** MLP vs the existing transformer encoder; ± FinBERT
  sentiment. Uninterpretable until the learning signal is fixed, so gated on
  A–C.
- **E — longer HPO proxy (I-8):** current proxy is 15 episodes
  (`tuning/best_config.json: tune_episodes = 15`); transfer to 200–500-ep runs
  is untested. Re-tuning after a reward-shape change is necessary anyway, so
  extend the proxy (e.g. 50–100 ep) at the same time.

---

## 5. Acceptance criteria (Phase 5, from handover §8.6)

- Documented, bounded α across all seeds/folds.
- Cost-aware reward implemented, leak-free, measured against the Phase 4 panel
  via the existing harnesses (no ad-hoc eval code).
- `make repro` passes.
- Honest framing preserved (per-fold/seed JK primary, DSR cross-check,
  net-of-cost always). Report the new gross→net gap and whether it closed.
- New unit tests for reward/entropy; full suite green (136+ tests).
- Authored Dhruvi, no co-author trailer.

## 6. Run recipe (zsh-safe, no inline comments)

```
conda activate portfolio-rl
cd ~/Documents/Projects/finrl/dynamic-portfolio-optimization
make test
make repro
caffeinate -i python experiments/multi_seed.py --seeds 0 1 2 3 4 --episodes 500 --config tuning/best_config.json
caffeinate -i python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 --min-train-months 12 --episodes 200 --config tuning/best_config.json
```

Budget: single-window ≈ 40h, walk-forward ≈ 40h+ on this MacBook Air (both
retrain the agent, unlike Phase 4's baseline-only work). `caffeinate -i` every
long run.

## 7. Out of scope (→ Phase 6)

README/doc rewrite, state-dim/FinRL claim fixes, repo-tree sync (I-9).

## 8. Order of work

C (cheap, no retrain — validate `make repro` first) → A (entropy, validate on
short runs before full retrain) → B (reward, the net-result lever; requires
re-tune per Task E) → E (longer proxy, folded into B's re-tune) → D (ablations).
