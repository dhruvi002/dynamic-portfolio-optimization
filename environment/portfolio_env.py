"""
Portfolio Trading Environment (FinRL-compatible)
==================================================
Wraps a multi-asset price dataset into a Gymnasium environment.

State space:
    [portfolio_weights(n), price_returns(n), tech_indicators(n × k)]

Action space:
    Continuous weights ∈ [0,1]^n  (agent output; env normalises to sum=1)

Reward:
    log(portfolio_return) − λ_tc * transaction_cost − λ_slip * slippage

Where:
    transaction_cost = tc_rate * sum(|Δw|) * portfolio_value
    slippage         = slip_rate * sum(|Δw|) * portfolio_value
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional

from config import UNIVERSE


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Leak-free trading universe (Phase 2, I-4). Single source of truth is
    # config.UNIVERSE — a fixed set of continuous Dow-30 members across the
    # 2018–2025 window, NOT the live DJ-30 (no survivorship/look-ahead bias).
    UNIVERSE = list(UNIVERSE)
    # Backward-compatible alias so existing references keep working; it now
    # resolves to the leak-free universe rather than the old back-filled DJ-30.
    DJ30_TICKERS = UNIVERSE

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: list = None,
        tech_indicators: list = None,
        initial_capital: float = 1_000_000.0,
        transaction_cost_rate: float = 0.001,   # 0.1 %
        slippage_rate: float = 0.001,            # 0.1 %
        reward_scaling: float = 1e-4,
        lookback: int = 1,
        seed: Optional[int] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()

        self.tickers = tickers or self.UNIVERSE
        self.n_assets = len(self.tickers)
        self.tech_indicators = tech_indicators or ["macd", "rsi_30", "cci_30", "dx_30"]
        self.n_tech = len(self.tech_indicators)

        self.initial_capital = initial_capital
        self.tc_rate = transaction_cost_rate
        self.slip_rate = slippage_rate
        self.reward_scaling = reward_scaling
        self.lookback = lookback

        # Filter to only dates where ALL tickers have data
        df = df.copy().sort_values(["date", "tic"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        tickers_in_df = df.groupby("date")["tic"].apply(set)
        required = set(self.tickers)
        valid_dates = tickers_in_df[tickers_in_df.apply(lambda s: required.issubset(s))].index
        self.df = df[df["date"].isin(valid_dates)].reset_index(drop=True)
        n_dropped = len(tickers_in_df) - len(valid_dates)
        if n_dropped > 0:
            print(f"  [Env] Dropped {n_dropped} dates missing one or more tickers. "
                  f"{len(valid_dates)} dates remaining.")

        # Build date index
        self.dates = sorted(self.df["date"].unique())
        self.n_steps = len(self.dates) - 1

        # Optional sentiment — pre-pivot for O(1) lookup during step()
        self.has_sentiment = sentiment_df is not None
        if self.has_sentiment:
            sdf = sentiment_df.copy()
            sdf["date"] = pd.to_datetime(sdf["date"])
            self._sentiment_pivot = (
                sdf.pivot_table(index="date", columns="tic", values="sentiment_score")
                   .reindex(columns=self.tickers)
                   .fillna(0.0)
            )

        # State: [weights(n), returns(n), tech(n × k), (sentiment(n) if enabled)]
        self.state_dim = (
            self.n_assets + self.n_assets + self.n_assets * self.n_tech
            + (self.n_assets if self.has_sentiment else 0)
        )
        self.action_dim = self.n_assets

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self.reset()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_prices(self, date_idx: int) -> np.ndarray:
        date = self.dates[date_idx]
        rows = self.df[self.df["date"] == date].set_index("tic")
        return rows.loc[self.tickers, "close"].values.astype(np.float32)

    def _get_tech(self, date_idx: int) -> np.ndarray:
        date = self.dates[date_idx]
        rows = self.df[self.df["date"] == date].set_index("tic")
        tech = rows.loc[self.tickers, self.tech_indicators].values.astype(np.float32)
        return tech.flatten()

    def _get_sentiment(self, date_idx: int) -> np.ndarray:
        date = self.dates[date_idx]
        if date in self._sentiment_pivot.index:
            return self._sentiment_pivot.loc[date].values.astype(np.float32)
        return np.zeros(self.n_assets, dtype=np.float32)

    def _build_state(self) -> np.ndarray:
        prices_now = self._get_prices(self.current_step)
        prices_prev = self._get_prices(max(self.current_step - 1, 0))
        returns = (prices_now - prices_prev) / (prices_prev + 1e-8)
        tech = self._get_tech(self.current_step)
        parts = [self.weights, returns, tech]
        if self.has_sentiment:
            parts.append(self._get_sentiment(self.current_step))
        return np.concatenate(parts).astype(np.float32)

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.prices = self._get_prices(0)
        self.history = {
            "portfolio_value": [self.initial_capital],
            "weights": [self.weights.copy()],
            "returns": [],
        }
        return self._build_state(), {}

    def step(self, action: np.ndarray):
        # Normalise action → valid portfolio weights
        action = np.clip(action, 0, 1)
        total = action.sum()
        if total < 1e-8:
            action = np.ones(self.n_assets) / self.n_assets
        else:
            action = action / total
        new_weights = action.astype(np.float32)

        # Transaction costs + slippage (both proportional to turnover)
        delta_weights = np.abs(new_weights - self.weights)
        turnover = delta_weights.sum()
        tc = self.tc_rate * turnover * self.portfolio_value
        slip = self.slip_rate * turnover * self.portfolio_value

        # Advance to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps

        new_prices = self._get_prices(self.current_step)
        price_returns = new_prices / (self.prices + 1e-8)          # per-asset multiplier
        port_return = float(np.dot(new_weights, price_returns))     # weighted return

        # New portfolio value after return, costs, slippage
        prev_value = self.portfolio_value          # capture BEFORE overwrite
        gross = self.portfolio_value * port_return
        net = gross - tc - slip
        self.portfolio_value = max(net, 1.0)  # floor at $1 to avoid log(0)

        # log(net / prev) — computed against prev_value, not the updated field
        log_return = np.log(max(net, 1e-8) / prev_value)
        reward = float(log_return * self.reward_scaling)

        self.weights = new_weights
        self.prices = new_prices

        self.history["portfolio_value"].append(self.portfolio_value)
        self.history["weights"].append(self.weights.copy())
        self.history["returns"].append(port_return - 1.0)

        obs = self._build_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "port_return": port_return - 1.0,
            "turnover": turnover,
            "tc": tc,
            "slip": slip,
        }
        return obs, reward, done, False, info

    def render(self):
        print(
            f"Step {self.current_step:4d} | "
            f"Value: ${self.portfolio_value:,.0f} | "
            f"Top-3 weights: {np.argsort(self.weights)[-3:][::-1]}"
        )
