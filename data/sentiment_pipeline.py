"""
Sentiment Pipeline
==================
Offline FinBERT sentiment scoring for DJ30 tickers using the FNSPID dataset.

Dataset:  "Financial News and Stock Price Integration Dataset" (FNSPID)
Paper:    https://arxiv.org/abs/2402.06698
HuggingFace: Zihan1004/FNSPID  (file: Stock_news/nasdaq_exteral_data.csv)

FNSPID column names (nasdaq_exteral_data.csv):
    Date            — "YYYY-MM-DD HH:MM:SS UTC"
    Article_title   — headline text  ← run FinBERT on this
    Stock_symbol    — ticker (e.g. "AAPL")
    Url, Publisher, Author, Article, Lsa_summary, …  (unused)

One-time precompute — use the ready-made script instead of calling this directly:
    /opt/anaconda3/envs/portfolio-rl/bin/python scripts/precompute_sentiment.py

Or call manually:
    from data.sentiment_pipeline import build_sentiment_df, load_precomputed, merge_sentiment
    import pandas as pd
    from transformers import pipeline as hf_pipeline

    news_df  = pd.read_csv("data/nasdaq_exteral_data.csv")
    finbert  = hf_pipeline("text-classification", model="ProsusAI/finbert",
                            truncation=True, max_length=128, device=-1)
    sent_df  = build_sentiment_df(news_df, finbert)
    sent_df.to_parquet("data/sentiment_scores.parquet", index=False)
    print(f"Saved {len(sent_df)} (date, ticker) rows")

Output sentiment_score: float ∈ (-1, 1)  (positive→+, negative→−, neutral→0)
"""

import os
import pandas as pd

DJ30_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "SHW", "WMT",
]


def _label_to_score(label: str, score: float) -> float:
    """Convert FinBERT label + confidence → signed scalar ∈ (-1, 1)."""
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0  # neutral


def build_sentiment_df(
    news_df: pd.DataFrame,
    finbert_pipeline,
    tickers: list = None,
    batch_size: int = 64,
    date_col: str = "Date",
    ticker_col: str = "Stock_symbol",
    title_col: str = "Article_title",
) -> pd.DataFrame:
    """
    Run FinBERT on FNSPID headlines and return daily sentiment per ticker.

    Filters to DJ30 tickers first (~30K rows from 1.4M total), so runtime
    on CPU is ~30–60 min rather than days.

    Args:
        news_df:          Raw FNSPID DataFrame (nasdaq_exteral_data.csv).
        finbert_pipeline: HuggingFace text-classification pipeline (ProsusAI/finbert).
        tickers:          Restrict to this list (defaults to DJ30_TICKERS).
        batch_size:       FinBERT inference batch size (64 is safe for 8 GB RAM).
        date_col / ticker_col / title_col: Column names — defaults match FNSPID schema.

    Returns:
        DataFrame with columns: date (datetime64), tic (str), sentiment_score (float).
        One row per (date, tic) — daily mean across all same-day headlines.
    """
    tickers = tickers or DJ30_TICKERS

    df = news_df[[date_col, ticker_col, title_col]].copy()
    # FNSPID dates are "YYYY-MM-DD HH:MM:SS UTC" — parse as UTC then strip tz
    dates = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df["date"] = dates.dt.tz_convert(None).dt.normalize()
    df = df.rename(columns={ticker_col: "tic", title_col: "headline"})
    df = df[df["tic"].isin(tickers)].dropna(subset=["headline"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"] != ""]

    if df.empty:
        raise ValueError(
            f"No DJ30 rows found. Check ticker_col='{ticker_col}'. "
            "FNSPID uses 'Stock_symbol'."
        )

    print(f"  Running FinBERT on {len(df):,} headlines for {df['tic'].nunique()} DJ30 tickers…")
    headlines = df["headline"].tolist()
    scores_raw = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        results = finbert_pipeline(batch)
        scores_raw.extend([_label_to_score(r["label"], r["score"]) for r in results])
        if (i // batch_size) % 50 == 0:
            pct = 100 * (i + len(batch)) / len(headlines)
            print(f"    {i + len(batch):,} / {len(headlines):,}  ({pct:.1f}%)")

    df = df.reset_index(drop=True)
    df["sentiment_score"] = scores_raw

    daily = (
        df.groupby(["date", "tic"])["sentiment_score"]
        .mean()
        .reset_index()
    )
    print(f"  Done. {len(daily):,} (date, ticker) rows.")
    return daily


def load_precomputed(path: str = "data/sentiment_scores.parquet") -> pd.DataFrame:
    """Load precomputed sentiment scores from parquet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Precomputed sentiment not found at '{path}'.\n"
            "Run: python scripts/precompute_sentiment.py"
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def merge_sentiment(
    df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Left-join precomputed daily sentiment into the price+indicator DataFrame.

    Missing (date, tic) pairs → fill_value (0.0 = neutral).
    No forward-fill to avoid any risk of look-ahead.

    Args:
        df:           Price+indicator df with columns date, tic, close, macd, …
        sentiment_df: Output of load_precomputed().
        fill_value:   Score when no article exists for that (date, tic).

    Returns:
        df with an additional 'sentiment_score' column.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    sentiment_df = sentiment_df[["date", "tic", "sentiment_score"]].copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    merged = df.merge(sentiment_df, on=["date", "tic"], how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(fill_value)
    return merged
