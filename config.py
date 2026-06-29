# Single source of truth for all date ranges and the trading universe.
# DOWNLOAD_START gives ~1 year of indicator warmup before TRAIN_START.

DOWNLOAD_START   = "2018-01-01"
TRAIN_START      = "2019-04-01"
TRAIN_PROPER_END = "2021-12-31"   # train window end when val is held out
VAL_START        = "2022-01-01"   # validation window start
TRAIN_END        = "2022-12-31"   # end of val window (= full train+val range)
TEST_START       = "2023-01-01"
TEST_END         = "2025-01-31"

# ── Trading universe (Phase 2, I-4: survivorship / look-ahead fix) ───────────────
# The original code used the *current* Dow-30 membership with Sherwin-Williams
# (SHW) back-filled to 2018. SHW only joined the DJIA in Nov 2024, so using it for
# the 2019–2022 training window is look-ahead bias; likewise, excluding names that
# were dropped from the index (because they underperformed) is survivorship bias.
#
# UNIVERSE is a DISCLOSED FIXED NEUTRAL UNIVERSE: the subset of liquid large-cap
# names that were CONTINUOUS Dow-30 members across the entire 2018-01 → 2025-01
# window. It is NOT the live DJ-30. Mid-sample joiners/leavers are excluded so the
# universe carries no foreknowledge of index changes. See EXCLUDED_FROM_DJ30.
UNIVERSE = [
    "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "GS", "HD", "IBM",
    "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV",
    "UNH", "V", "VZ", "WMT",
]  # 24 names, continuous DJIA members 2018-01 → 2025-01

# Names dropped from the old hard-coded list, with the reason each is look-ahead /
# survivorship contaminated (documented per the Phase 2 acceptance criteria):
EXCLUDED_FROM_DJ30 = {
    "AMGN": "joined DJIA 2020-08-31 (not a member during 2019–mid-2020 training)",
    "CRM":  "joined DJIA 2020-08-31 (not a member during 2019–mid-2020 training)",
    "HON":  "joined DJIA 2020-08-31 (not a member during 2019–mid-2020 training)",
    "DOW":  "joined 2019-04-02, removed 2024-11-08 (not continuous across window)",
    "INTC": "removed from DJIA 2024-11-08 (not a member through window end)",
    "SHW":  "joined DJIA 2024-11-08 (back-filling to 2018 is look-ahead bias)",
}
