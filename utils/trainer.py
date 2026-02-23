"""
Training & backtesting utilities.
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional
import json
import os


def train(
    agent,
    env,
    n_episodes: int = 100,
    warmup_steps: int = 1000,
    update_every: int = 1,
    log_every: int = 10,
    save_path: Optional[str] = None,
    writer=None,                    # optional TensorBoard SummaryWriter
) -> list:
    """
    Full training loop.

    Returns list of per-episode metrics dicts.
    """
    episode_logs = []

    # ── Warm-up: fill replay buffer with random transitions ─────────────────
    print(f"Warming up replay buffer ({warmup_steps} random steps)…")
    state, _ = env.reset()
    step = 0
    while step < warmup_steps:
        action = env.action_space.sample()
        next_state, reward, done, _, info = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        state = next_state if not done else env.reset()[0]
        step += 1
    print(f"  Buffer size: {len(agent.replay_buffer)}")

    # ── Main training loop ───────────────────────────────────────────────────
    best_sharpe = -np.inf

    for ep in tqdm(range(1, n_episodes + 1), desc="Training"):
        state, _ = env.reset()
        done = False
        ep_returns, ep_losses = [], []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            if agent.total_steps % update_every == 0:
                loss_info = agent.update()
                if loss_info:
                    ep_losses.append(loss_info)

            ep_returns.append(info["port_return"])
            state = next_state

        returns_arr = np.array(ep_returns)
        port_values = np.array(env.history["portfolio_value"])

        from utils.metrics import compute_sharpe, compute_max_drawdown
        sharpe = compute_sharpe(returns_arr)
        mdd = compute_max_drawdown(port_values)
        mean_loss = {k: np.mean([l[k] for l in ep_losses]) for k in ep_losses[0]} if ep_losses else {}

        log = {
            "episode": ep,
            "total_return": float(port_values[-1] / port_values[0] - 1),
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "alpha": agent.alpha,
            **mean_loss,
        }
        episode_logs.append(log)

        if writer:
            for k, v in log.items():
                if k != "episode":
                    writer.add_scalar(f"train/{k}", v, ep)

        if ep % log_every == 0:
            print(
                f"  Ep {ep:4d} | Return: {log['total_return']:+.2%} | "
                f"Sharpe: {sharpe:.3f} | MDD: {mdd:.2%} | α: {agent.alpha:.4f}"
            )

        if save_path and sharpe > best_sharpe:
            best_sharpe = sharpe
            agent.save(save_path)
            print(f"  ✓ New best checkpoint (Sharpe={sharpe:.3f}) → {save_path}")

    return episode_logs


def backtest(agent, env) -> dict:
    """
    Run a single deterministic episode and return metrics.
    """
    from utils.metrics import compute_all_metrics

    state, _ = env.reset()
    done = False
    all_returns = []

    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, _, info = env.step(action)
        all_returns.append(info["port_return"])
        state = next_state

    port_values = np.array(env.history["portfolio_value"])
    returns_arr = np.array(all_returns)
    metrics = compute_all_metrics(returns_arr, port_values)
    metrics["final_value"] = float(port_values[-1])
    return metrics
