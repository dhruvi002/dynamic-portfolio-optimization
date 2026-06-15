#!/usr/bin/env python3
"""
One-time sentiment precompute using FNSPID + FinBERT.

Run once before training:
    /opt/anaconda3/envs/portfolio-rl/bin/python scripts/precompute_sentiment.py

Output: data/sentiment_scores.parquet
  Columns: date (datetime64), tic (str), sentiment_score (float ∈ -1..1)
  One row per (date, ticker) — daily mean over all same-day headlines.

Runtime: ~30–60 min on CPU (filters to DJ30 first, ~30K headlines from 1.4M total).
"""

import sys
import os

# Run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

NEWS_CSV   = "data/nasdaq_exteral_data.csv"
OUTPUT     = "data/sentiment_scores.parquet"
BATCH_SIZE = 64


def main():
    # ── 1. Check input ────────────────────────────────────────────────────────
    if not os.path.exists(NEWS_CSV):
        print(f"ERROR: {NEWS_CSV} not found.")
        print("Download it with:")
        print("  curl -L https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv -o data/nasdaq_exteral_data.csv")
        sys.exit(1)

    if os.path.exists(OUTPUT):
        print(f"Output already exists at {OUTPUT}. Delete it to rerun.")
        sys.exit(0)

    # ── 2. Load FinBERT ───────────────────────────────────────────────────────
    print("Loading ProsusAI/finbert (downloads ~500 MB on first run)…")
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("ERROR: transformers not installed.")
        print("  /opt/anaconda3/envs/portfolio-rl/bin/pip install transformers sentencepiece")
        sys.exit(1)

    finbert = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=128,   # headlines are short; 128 is enough and 4× faster than 512
        device=-1,        # CPU; change to device=0 if you have a GPU
    )

    # ── 3. Load news CSV ──────────────────────────────────────────────────────
    print(f"Loading {NEWS_CSV} (3.9 GB — using python engine to handle malformed rows)…")
    news_df = pd.read_csv(
        NEWS_CSV,
        usecols=["Date", "Stock_symbol", "Article_title"],
        engine="python",
        on_bad_lines="skip",
    )
    print(f"  {len(news_df):,} rows loaded")

    # ── 4. Run pipeline ───────────────────────────────────────────────────────
    from data.sentiment_pipeline import build_sentiment_df
    sentiment_df = build_sentiment_df(news_df, finbert, batch_size=BATCH_SIZE)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    sentiment_df.to_parquet(OUTPUT, index=False)
    print(f"\nSaved → {OUTPUT}")
    print(f"  {len(sentiment_df):,} (date, ticker) rows")
    print(f"  Date range: {sentiment_df['date'].min().date()} → {sentiment_df['date'].max().date()}")
    print(f"  Tickers:    {sorted(sentiment_df['tic'].unique())}")


if __name__ == "__main__":
    main()
