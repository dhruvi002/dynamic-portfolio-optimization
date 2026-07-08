# Deep RL Portfolio Optimization

A Soft Actor-Critic (SAC) agent with a Dirichlet policy that allocates a fixed
24-asset large-cap US equity portfolio, evaluated under a reproducible,
leakage-free, multi-seed, multi-regime, **cost-inclusive** protocol against a
panel of standard benchmarks.

Built with PyTorch, a hand-rolled FinRL-*compatible* Gymnasium environment
(not FinRL-powered), and a leak-free hyperparameter search. Trained and
backtested on real market data (2019–2025), CPU-only.

> **Result in one line:** net of realistic transaction costs, the agent does
> **not** beat any standard benchmark. Its gross (pre-cost) Sharpe is
> competitive, but turnover cost erases the edge; its best-behaving runs
> converge toward equal-weight and cannot beat it. This is a negative result,
> established rigorously — it replaces an earlier unseeded, leaky backtest that
> had claimed "+24% Sharpe / +47% return." Full detail in
> [`RESULTS.md`](RESULTS.md).

---

## Results (net of cost)

Test window **Jan 2023 – Jan 2025**, 5 seeds × 500 episodes. Sharpe is
value-path, net of 0.1% transaction + 0.1% slippage cost, applied identically
to the agent and every baseline.

| Strategy | Net Sharpe |
|---|---|
| SPY/QQQ 60/40 | +1.98 |
| SPY Buy & Hold | +1.92 |
| 60/40 SPY/AGG | +1.87 |
| Risk Parity | +1.69 |
| **Equal Weight** | **+1.63** |
| Rolling MVO-LW (max-Sharpe) | +1.61 |
| Max-Sharpe MVO (static) | +1.46 |
| Rolling MVO-LW (min-var) | +1.34 |
| Min Variance (static) | +1.05 |
| **SAC agent (entropy-fixed)** | **+0.84 ± 0.60** (gross +1.62) |
| Momentum 12-1 | +0.89 |

The agent's gross Sharpe (~+1.6) is in the benchmark range; a ~0.6–1.0
transaction-cost drag from turnover is the entire gap. It sits above the two
weakest baselines and below equal-weight and every strong benchmark.

**Walk-forward (27 folds across COVID / 2021 bull / 2022 bear / 2023-24
recovery):** overall net Sharpe **+0.59 ± 1.54**, beats equal-weight in
**0/27 folds**. See [`RESULTS.md`](RESULTS.md) for per-regime tables and
significance.

Figures: `experiments/results/walk_forward_baseline_panel.png` (agent vs full
panel, per regime) and `experiments/results/walk_forward_regimes.png`.

---

## Why Soft Actor-Critic?

Portfolio allocation is a continuous-control problem — the agent outputs a
weight vector over 24 assets every step. SAC fits because:

- **Maximum-entropy objective** — optimizes reward *and* policy entropy, which
  discourages overcommitting to one allocation and encourages diversification.
- **Automatic entropy tuning** — the temperature α is learned. (See the
  Limitations section: on this problem α saturates at its clamp, effectively
  acting as a fixed strong entropy regularizer.)
- **Twin critics** — two Q-networks reduce Q-value overestimation, important in
  a domain where overconfidence is costly.
- **Off-policy** — a replay buffer makes it sample-efficient on CPU.

The policy is a **Dirichlet distribution on the K-simplex**, so samples are
valid portfolio weights (non-negative, sum to 1) with an exact log-density — no
softmax/Jacobian approximation.

---

## Architecture

```
dynamic-portfolio-optimization/
├── agent/
│   └── sac.py                    # SAC: DirichletActor, twin Critics, ReplayBuffer,
│                                 #   bounded auto entropy tuning, optional transformer encoder
├── environment/
│   └── portfolio_env.py          # FinRL-compatible Gymnasium env (24 assets, costs, slippage)
├── data/
│   └── pipeline.py               # yfinance download + MACD, RSI, CCI, ADX features
├── tuning/
│   └── tune_runner.py            # leak-free HPO (Ray Tune w/ Ray-free fallback)
├── utils/
│   ├── metrics.py                # Sharpe, Sortino, Calmar, max drawdown
│   ├── baselines.py              # equal-weight, SPY B&H, 60/40, risk parity, rolling MVO-LW, …
│   ├── walk_forward.py           # expanding-window, net-of-cost, leak-guarded
│   ├── significance.py           # Jobson–Korkie–Memmel, bootstrap CI, PSR/DSR/PBO
│   ├── seeding.py                # global seeding + CPU determinism (thread pinning)
│   ├── trainer.py                # training loop + backtest runner
│   └── plotting.py               # portfolio value, drawdown, weight/panel figures
├── experiments/
│   ├── multi_seed.py             # single-window multi-seed harness (make evaluate)
│   ├── walk_forward_eval.py      # multi-regime walk-forward harness (make walkforward)
│   └── compare_sweep.py          # side-by-side sweep comparison
├── test/                         # pytest suite incl. leak / walk-forward / baseline guards
├── config.py                     # date windows + 24-name UNIVERSE
└── main.py                       # CLI: train / tune / backtest / walkforward
```

**State space (144-dim):** portfolio weights (24) + daily returns (24) +
technical indicators (24 × 4: MACD, RSI, CCI, ADX).

**Action space (24-dim):** portfolio weights on the simplex (Dirichlet policy;
the env also clips/normalizes any raw action to sum to 1).

**Reward:** `(log(net_value / prev_value) − λ_turnover · Σ|Δw|) · reward_scaling`.
The log-return term is already net of transaction + slippage cost; `λ_turnover`
(default 0) adds an optional explicit turnover penalty.

---

## Quickstart

### 1. Install
```bash
conda env create -f environment.yml    # creates the `portfolio-rl` env (Python 3.10, CPU)
conda activate portfolio-rl
```

### 2. Reproducibility check
```bash
make test        # full unit suite
make repro       # trains twice at one seed; asserts identical logs
```

### 3. Train
```bash
python main.py --mode train --episodes 500 --config tuning/best_config.json
```
Downloads the 24-name universe (2019–2025), computes indicators, trains SAC,
saves the best-by-validation-Sharpe checkpoint to `checkpoints/best_agent.pt`.

### 4. Evaluate (multi-seed + walk-forward harnesses)
```bash
# single-window, multi-seed, vs the full baseline panel
python experiments/multi_seed.py --seeds 0 1 2 3 4 --episodes 500 --config tuning/best_config.json

# multi-regime walk-forward
python experiments/walk_forward_eval.py --seeds 0 1 2 --folds 9 --test-months 6 \
    --min-train-months 12 --episodes 200 --config tuning/best_config.json
```

Optional knobs: `--turnover-penalty <λ>` and `--reward-scaling <s>` (both were
swept; see [`RESULTS.md`](RESULTS.md)).

---

## Environment details

| Parameter | Value |
|---|---|
| Assets | 24-name fixed neutral universe (see Leakage removal below) |
| State / action dim | 144 / 24 |
| Training period | Apr 2019 – Dec 2021 (train) / 2022 (validation) |
| Test period | Jan 2023 – Jan 2025 |
| Transaction cost | 0.1% per unit turnover |
| Slippage | 0.1% per unit turnover |
| Initial capital | $1,000,000 |
| Rebalancing | Daily |

---

## Evaluation rigor

- **Reproducible:** global seeding, version-pinned deps, run metadata stamped;
  `make repro` gives identical logs for identical seeds (CPU threads pinned).
- **Leakage-free:** the HPO no longer sees the test set, and the universe is a
  disclosed fixed set with no look-ahead (see below). Guarded by
  `test/test_no_leak.py`, `test/test_walk_forward.py`, `test/test_baselines.py`.
- **Cost-inclusive:** every headline metric is net of cost via the value path;
  gross is reported alongside to isolate the turnover drag.
- **Statistically stated:** per-seed / per-(seed,fold) Jobson–Korkie–Memmel is
  primary; pooled JK, bootstrap CIs, and the Deflated Sharpe Ratio are
  cross-checks. Results span four market regimes over 27 walk-forward folds.

### Leakage removal (I-3, I-4)

**I-3 — HPO no longer sees the test set.** Each tuning trial does a chronological
`three_way_split`, trains on train only, and scores on the deterministic,
net-of-cost **validation** Sharpe. Seeded and guarded by `test/test_no_leak.py`.

**I-4 — survivorship / look-ahead universe.** The universe is a disclosed fixed
set (`config.UNIVERSE`): the **24 names that were continuous DJIA members across
2018-01 → 2025-01** — not the live DJ-30. Excluded mid-sample joiners/leavers:

| Excluded | Reason |
|---|---|
| AMGN, CRM, HON | joined the DJIA 2020-08-31 |
| DOW | joined 2019-04-02, removed 2024-11-08 (not continuous) |
| INTC | removed 2024-11-08 |
| SHW | joined 2024-11-08 (back-filling to 2018 is look-ahead) |

The excluded tickers are listed above; the leakage-free universe is defined in
`config.UNIVERSE`.

---

## Limitations

- **No net-of-cost edge.** The agent does not beat equal-weight or any strong
  benchmark after costs (0/27 walk-forward folds). Its active tilts are
  value-destroying net of cost; the best it does is rediscover diversification.
- **Seed instability.** Per-seed net Sharpe spans ~0.1 to ~1.5 depending on
  which basin a seed lands in; 3–5 seeds bound but do not remove this.
- **No result survives the multiple-testing haircut** (Deflated Sharpe ≈ 0).
- **α saturates** at its clamp ceiling across every tested setting — the clamp
  is a fixed strong regularizer, not a tuned quantity.
- **HPO transfer gap.** Hyperparameters were tuned with a short (15-episode)
  proxy; transfer to 500-episode runs is not validated, and the reward-shape
  change argues for a re-tune not yet done.

---

## Tech stack

| Component | Library |
|---|---|
| RL agent | PyTorch (custom SAC + Dirichlet policy) |
| Environment | Gymnasium (hand-rolled, FinRL-compatible; FinRL not used) |
| Data | yfinance |
| Technical indicators | ta |
| Baselines / stats | NumPy, pandas, scikit-learn (Ledoit-Wolf) |
| Hyperparameter tuning | Ray Tune + HyperOpt (Ray-free fallback) |
| Experiment tracking | TensorBoard |

---

## Project history

The project was remediated in phases: reproducibility → leakage removal →
multi-regime evaluation → strong baseline panel → entropy fix + cost-aware
reward. The consolidated result is in [`RESULTS.md`](RESULTS.md).
