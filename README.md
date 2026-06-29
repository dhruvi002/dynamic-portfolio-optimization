# Deep RL Portfolio Optimization

A self-optimizing trading agent using **Soft Actor-Critic (SAC)** to dynamically manage a 30-asset Dow Jones portfolio, achieving a **+24% improvement in Sharpe Ratio** over an equal-weight baseline.

Built with PyTorch, FinRL, and Ray Tune. Trained and backtested on real market data (2019–2025).

---

## Results

Backtested on out-of-sample data from **January 2023 – January 2025**:

| Metric | SAC Agent | Equal-Weight Baseline | Δ |
|---|---|---|---|
| **Sharpe Ratio** | **1.863** | 1.504 | **+24%** |
| **Sortino Ratio** | **2.977** | 2.297 | **+30%** |
| **Calmar Ratio** | **3.010** | 2.122 | **+42%** |
| **Max Drawdown** | **-6.9%** | -8.4% | **+1.5%** |
| **Total Return** | **47.3%** | 40.6% | **+6.7%** |
| **Ann. Return** | **20.8%** | 17.8% | **+3.0%** |
| **Ann. Volatility** | **10.1%** | 10.9% | **lower** |
| **Final Value** | **$1,472,567** | $1,405,453 | **+$67,114** |

> Starting capital: $1,000,000. Transaction costs: 0.1%. Slippage: 0.1%.

---

## Plots

### Portfolio Value vs Equal-Weight Baseline
![Backtest Comparison](plots/backtest_comparison.png)

### Training Curves (500 Episodes)
![Training Curves](plots/training_curves.png)

### Portfolio Weight Allocation Over Time
![Weight Heatmap](plots/weight_heatmap.png)

---

## Why Soft Actor-Critic?

Portfolio optimization is a continuous control problem — the agent must output a weight vector across 30 assets at every timestep. SAC is well-suited for this because:

- **Maximum entropy framework** — SAC optimizes for both reward *and* policy entropy, which prevents overcommitting to a single asset allocation and naturally encourages diversification
- **Automatic entropy tuning** — the temperature parameter α is learned during training, automatically balancing exploration vs exploitation without manual tuning
- **Twin critics** — two Q-networks trained in parallel reduce Q-value overestimation, leading to more stable and conservative policy updates — important in financial environments where overconfidence is costly
- **Off-policy** — SAC reuses past experience via a replay buffer, making it sample-efficient even on CPU

---

## Architecture

```
portfolio_rl/
├── agent/
│   └── sac.py              # SAC: Actor, Twin Critics, ReplayBuffer, auto entropy tuning
├── environment/
│   └── portfolio_env.py    # FinRL-compatible Gymnasium env (30 assets, costs, slippage)
├── data/
│   └── pipeline.py         # yfinance download + MACD, RSI, CCI, ADX features
├── tuning/
│   └── tune_runner.py      # Ray Tune ASHA scheduler + HyperOpt TPE (50 trials)
├── utils/
│   ├── metrics.py          # Sharpe, Sortino, Calmar, Max Drawdown
│   ├── trainer.py          # Training loop + backtest runner
│   └── plotting.py         # Portfolio value, drawdown, weight heatmap plots
├── checkpoints/            # Saved model weights + training logs
├── plots/                  # Generated figures
└── main.py                 # CLI: train / tune / backtest
```

**State space** (180-dim): portfolio weights (30) + daily returns (30) + technical indicators (30 × 4: MACD, RSI, CCI, ADX)

**Action space** (30-dim): continuous portfolio weights ∈ [0,1], normalized to sum to 1 via softmax

**Reward**: log portfolio return − transaction cost penalty − slippage penalty

---

## Quickstart

### 1. Install dependencies
```bash
conda create -n portfolio-rl python=3.11
conda activate portfolio-rl
pip install -r requirements.txt
```

### 2. Train
```bash
python main.py --mode train --episodes 500
```
Downloads DJ30 data (2019–2025), computes indicators, trains SAC for 500 episodes (~20 min on CPU). Saves best checkpoint to `checkpoints/best_agent.pt`.

### 3. Hyperparameter optimization (optional)
```bash
python main.py --mode tune --tune-samples 50
```
Runs 50 Ray Tune trials with ASHA early stopping and HyperOpt TPE search across learning rates, gamma, tau, batch size, and network width. Saves best config to `tuning/best_config.json`.

### 4. Backtest
```bash
python main.py --mode backtest --checkpoint checkpoints/best_agent.pt
```
Runs deterministic rollout on held-out test data and prints full metrics table. Saves plots to `plots/`.

---

## Environment Details

| Parameter | Value |
|---|---|
| Assets | 24-name fixed neutral universe (see Phase 2 below) |
| Training period | Apr 2019 – Dec 2021 (train) / 2022 (validation) |
| Test period | Jan 2023 – Jan 2025 |
| Transaction cost | 0.1% per turnover |
| Slippage | 0.1% per turnover |
| Initial capital | $1,000,000 |
| Rebalancing | Daily |

---

## Phase 2 — Leakage removal (I-3, I-4)

> The original headline ("+24% Sharpe") rested on a single unseeded backtest **with
> two leaks**. Phases 0–1 made evaluation reproducible and CI/significance-backed;
> Phase 2 removes the leaks so the numbers can be trusted, then re-runs the
> unchanged Phase 1 harness. Results are being re-measured under the corrected
> pipeline — treat the legacy "+24%" numbers above as superseded.

**I-3 — HPO no longer sees the test set.** The Ray Tune trial path
(`tuning/tune_runner.py`) previously trained and scored on the *entire*
2019–2025 series (including the 2023–2025 test window), so the test set influenced
the selected hyperparameters. Each trial now does a chronological
`three_way_split`, trains on **train only**, and reports the **deterministic,
net-of-cost validation Sharpe** as its objective — matching the metric the Phase 1
harness reports. Trials are seeded for reproducibility, and a hard guard
(`test/test_no_leak.py`) asserts no test-window row ever enters tuning. The old
hyperparameters are archived at `tuning/best_config_LEAKED.json`.

**I-4 — survivorship / look-ahead universe.** The universe was the *current*
Dow-30 with Sherwin-Williams (SHW) back-filled to 2018; SHW only joined the DJIA
in Nov 2024. The trading universe is now a **disclosed fixed neutral set**
(`config.UNIVERSE`): the **24 names that were continuous DJIA members across the
entire 2018-01 → 2025-01 window** — not the live DJ-30, and carrying no
foreknowledge of index changes. Excluded mid-sample joiners/leavers:

| Excluded | Reason |
|---|---|
| AMGN, CRM, HON | joined the DJIA 2020-08-31 (not members during 2019–mid-2020 training) |
| DOW | joined 2019-04-02, removed 2024-11-08 (not continuous across the window) |
| INTC | removed from the DJIA 2024-11-08 (not a member through the window end) |
| SHW | joined the DJIA 2024-11-08 (back-filling to 2018 is look-ahead bias) |

See `PHASE2_NOTES.md` for the full rationale and the re-measurement procedure.

---

## Tech Stack

| Component | Library |
|---|---|
| RL Agent | PyTorch (custom SAC) |
| Trading Environment | FinRL / Gymnasium |
| Data | yfinance |
| Technical Indicators | ta |
| Hyperparameter Tuning | Ray Tune + HyperOpt |
| Experiment Tracking | TensorBoard |
