"""
Market-regime labelling for the Phase-3 walk-forward evaluation.
================================================================
Each walk-forward fold's *test window* is mapped to a named market regime so the
per-regime table can state where the agent does best / worst. A fold is assigned
by the **midpoint** of its test window (start + (end-start)/2); this is robust to
the 6-month folds straddling a calendar boundary.

The cutoffs below are documented and fixed. They cover the span the 24-name
universe trades over (≈2019-04 → 2025-01) and are chosen to separate the four
qualitatively different regimes the handover calls out:

    COVID crash/recovery   2020-02-01 → 2020-12-31   (Feb–Mar crash + V-recovery)
    2021 bull              2021-01-01 → 2021-12-31   (low-rate melt-up)
    2022 bear (rate shock) 2022-01-01 → 2022-12-31   (Fed hiking, multiple compression)
    2023–24 recovery/AI    2023-01-01 → 2025-01-31   (disinflation + AI rally)

A pre-COVID label is provided for completeness in case a smaller
``min_train_months`` pushes a fold before 2020-02; folds whose midpoint predates
all ranges fall back to ``"pre-COVID 2019"``.
"""

import pandas as pd

# (label, inclusive_start, inclusive_end) — ordered, non-overlapping.
REGIME_BANDS = [
    ("pre-COVID 2019",          "2019-01-01", "2020-01-31"),
    ("COVID crash/recovery",    "2020-02-01", "2020-12-31"),
    ("2021 bull",               "2021-01-01", "2021-12-31"),
    ("2022 bear (rate shock)",  "2022-01-01", "2022-12-31"),
    ("2023-24 recovery/AI",     "2023-01-01", "2025-12-31"),
]

# Display order for tables/figures.
REGIME_ORDER = [b[0] for b in REGIME_BANDS]


def regime_for(test_start, test_end) -> str:
    """
    Return the regime label for a fold given its test-window endpoints
    (date strings or Timestamps). Assignment is by the window midpoint.
    """
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    mid = ts + (te - ts) / 2
    for label, lo, hi in REGIME_BANDS:
        if pd.Timestamp(lo) <= mid <= pd.Timestamp(hi):
            return label
    # Midpoint past the last band's end (shouldn't happen for this universe).
    return REGIME_BANDS[-1][0] if mid > pd.Timestamp(REGIME_BANDS[-1][2]) \
        else REGIME_BANDS[0][0]


def regimes_present(fold_results) -> list:
    """Distinct regime labels appearing in a list of fold dicts, in display order."""
    present = {regime_for(f["test_start"], f["test_end"]) for f in fold_results}
    return [r for r in REGIME_ORDER if r in present]
