"""
Data Pipeline
=============
Downloads DJ30 OHLCV data via yfinance and computes technical indicators
(MACD, RSI, CCI, ADX) using the `ta` library — mirroring FinRL's preprocessing.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

DJ30_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "SHW", "WMT",
]  # WBA delisted -> replaced with SHW (Sherwin-Williams)

INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]


def download_data(
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    tickers: list = None,
    cache_path: str = "data/raw_data.parquet",
) -> pd.DataFrame:
    tickers = tickers or DJ30_TICKERS
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")

    all_dfs = []
    for tic in tickers:
        try:
            raw = yf.download(tic, start=start, end=end, auto_adjust=True,
                              progress=False, multi_level_index=False)
            if raw.empty:
                print(f"  WARNING {tic}: empty dataframe, skipping")
                continue
            raw = raw.reset_index()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0].lower() for col in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw["tic"] = tic
            if "datetime" in raw.columns:
                raw = raw.rename(columns={"datetime": "date"})
            # Keep as datetime (not .date()) so resample works
            raw["date"] = pd.to_datetime(raw["date"])
            all_dfs.append(raw[["date", "tic", "open", "high", "low", "close", "volume"]])
            print(f"  OK {tic} ({len(raw)} rows)")
        except Exception as e:
            print(f"  WARNING {tic}: {e}")

    if not all_dfs:
        raise RuntimeError("No ticker data downloaded.")

    df = pd.concat(all_dfs, ignore_index=True).sort_values(["date", "tic"])

    # Forward-fill missing business days — requires DatetimeIndex
    df = (
        df.groupby("tic", group_keys=False)
          .apply(lambda g: g.set_index("date").resample("B").ffill().reset_index())
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date

    try:
        df.to_parquet(cache_path, index=False)
        print(f"  Saved to {cache_path}")
    except Exception:
        pass

    print(f"  Total: {len(df)} rows, {df['tic'].nunique()} tickers")
    return df.reset_index(drop=True)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing technical indicators...")
    result_dfs = []

    for tic in df["tic"].unique():
        sub = df[df["tic"] == tic].copy().sort_values("date")

        macd_obj = ta.trend.MACD(sub["close"])
        sub["macd"] = macd_obj.macd_diff().fillna(0)
        sub["rsi_30"] = ta.momentum.RSIIndicator(sub["close"], window=30).rsi().fillna(50)
        sub["cci_30"] = ta.trend.CCIIndicator(
            sub["high"], sub["low"], sub["close"], window=30
        ).cci().fillna(0)
        sub["dx_30"] = ta.trend.ADXIndicator(
            sub["high"], sub["low"], sub["close"], window=30
        ).adx().fillna(0)

        result_dfs.append(sub)

    result = pd.concat(result_dfs, ignore_index=True)
    print(f"  Indicators added. Shape: {result.shape}")
    return result.sort_values(["date", "tic"]).reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    train_start: str = "2019-04-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2025-01-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df["date"] = pd.to_datetime(df["date"])
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    print(f"  Train: {train['date'].min().date()} -> {train['date'].max().date()} "
          f"({len(train['date'].unique())} trading days)")
    print(f"  Test:  {test['date'].min().date()} -> {test['date'].max().date()} "
          f"({len(test['date'].unique())} trading days)")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def three_way_split(
    df: pd.DataFrame,
    train_start: str = "2019-04-01",
    train_end: str = "2021-12-31",
    val_start: str = "2022-01-01",
    val_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2025-01-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / val / test split with no overlapping windows."""
    df["date"] = pd.to_datetime(df["date"])
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    val   = df[(df["date"] >= val_start)   & (df["date"] <= val_end)].copy()
    test  = df[(df["date"] >= test_start)  & (df["date"] <= test_end)].copy()
    print(f"  Train: {train['date'].min().date()} -> {train['date'].max().date()} "
          f"({len(train['date'].unique())} trading days)")
    print(f"  Val:   {val['date'].min().date()} -> {val['date'].max().date()} "
          f"({len(val['date'].unique())} trading days)")
    print(f"  Test:  {test['date'].min().date()} -> {test['date'].max().date()} "
          f"({len(test['date'].unique())} trading days)")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
