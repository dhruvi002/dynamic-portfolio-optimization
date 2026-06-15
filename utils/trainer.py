"""
Training & backtesting utilities.
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional
import json
import os


def _norm(normalizer, state, update: bool = False):
    """Apply normalizer if provided, optionally updating its running stats."""
    if normalizer is None:
        return state
    if update:
        normalizer.update(state)
    return normalizer.normalize(state)


def train(
    agent,
    env,
    n_episodes: int = 100,
    warmup_steps: int = 1000,
    update_every: int = 1,
    log_every: int = 10,
    save_path: Optional[str] = None,
    writer=None,
    normalizer=None,
    val_env=None,
) -> list:
    """
    Full training loop.

    val_env: if provided, runs a deterministic backtest on val_env after each
    episode and uses val_sharpe (not train_sharpe) for checkpoint selection.
    This prevents the model from over-fitting to training-period performance.

    Returns list of per-episode metrics dicts.
    """
    if normalizer is not None:
        normalizer.train()

    episode_logs = []

    # ── Warm-up: fill replay buffer with random transitions ─────────────────
    print(f"Warming up replay buffer ({warmup_steps} random steps)…")
    state, _ = env.reset()
    step = 0
    while step < warmup_steps:
        norm_state = _norm(normalizer, state, update=True)
        action     = env.action_space.sample()
        next_state, reward, done, _, info = env.step(action)
        norm_next  = _norm(normalizer, next_state, update=False)
        agent.replay_buffer.push(norm_state, action, reward, norm_next, float(done))
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
            norm_state = _norm(normalizer, state, update=True)
            action     = agent.select_action(norm_state)
            next_state, reward, done, _, info = env.step(action)
            norm_next  = _norm(normalizer, next_state, update=False)
            agent.replay_buffer.push(norm_state, action, reward, norm_next, float(done))

            if agent.total_steps % update_every == 0:
                loss_info = agent.update()
                if loss_info:
                    ep_losses.append(loss_info)

            ep_returns.append(info["port_return"])
            state = next_state

        returns_arr  = np.array(ep_returns)
        port_values  = np.array(env.history["portfolio_value"])

        from utils.metrics import compute_sharpe, compute_max_drawdown
        sharpe   = compute_sharpe(returns_arr)
        mdd      = compute_max_drawdown(port_values)
        mean_loss = {k: np.mean([l[k] for l in ep_losses]) for k in ep_losses[0]} if ep_losses else {}

        log = {
            "episode":      ep,
            "total_return": float(port_values[-1] / port_values[0] - 1),
            "sharpe":       sharpe,
            "max_drawdown": mdd,
            "alpha":        agent.alpha,
            **mean_loss,
        }

        # ── Validation pass ──────────────────────────────────────────────────
        val_sharpe = None
        if val_env is not None:
            val_metrics = backtest(agent, val_env, normalizer=normalizer)
            val_sharpe = val_metrics["sharpe"]
            if normalizer is not None:
                normalizer.train()  # restore train mode after val backtest
            log["val_sharpe"] = val_sharpe
            if writer:
                writer.add_scalar("val/sharpe", val_sharpe, ep)

        episode_logs.append(log)

        if writer:
            for k, v in log.items():
                if k not in ("episode", "val_sharpe"):
                    writer.add_scalar(f"train/{k}", v, ep)

        if ep % log_every == 0:
            val_str = f" | Val Sharpe: {val_sharpe:.3f}" if val_sharpe is not None else ""
            print(
                f"  Ep {ep:4d} | Return: {log['total_return']:+.2%} | "
                f"Sharpe: {sharpe:.3f} | MDD: {mdd:.2%} | α: {agent.alpha:.4f}{val_str}"
            )

        # Checkpoint on val_sharpe when available, else train_sharpe
        checkpoint_metric = val_sharpe if val_sharpe is not None else sharpe
        if save_path and checkpoint_metric > best_sharpe:
            best_sharpe = checkpoint_metric
            agent.save(save_path)
            metric_label = "Val Sharpe" if val_sharpe is not None else "Sharpe"
            print(f"  ✓ New best checkpoint ({metric_label}={checkpoint_metric:.3f}) → {save_path}")

    return episode_logs


def backtest(agent, env, normalizer=None) -> dict:
    """
    Run a single deterministic episode and return metrics.
    Normalizer is frozen (eval mode) so test data cannot shift training stats.
    """
    from utils.metrics import compute_all_metrics

    if normalizer is not None:
        normalizer.eval()

    state, _ = env.reset()
    done      = False
    all_returns = []

    while not done:
        norm_state = _norm(normalizer, state, update=False)
        action     = agent.select_action(norm_state, deterministic=True)
        next_state, reward, done, _, info = env.step(action)
        all_returns.append(info["port_return"])
        state = next_state

    port_values  = np.array(env.history["portfolio_value"])
    returns_arr  = np.array(all_returns)
    metrics      = compute_all_metrics(returns_arr, port_values)
    metrics["final_value"] = float(port_values[-1])
    return metrics
