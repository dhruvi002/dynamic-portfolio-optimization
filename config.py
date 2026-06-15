# Single source of truth for all date ranges.
# DOWNLOAD_START gives ~1 year of indicator warmup before TRAIN_START.

DOWNLOAD_START   = "2018-01-01"
TRAIN_START      = "2019-04-01"
TRAIN_PROPER_END = "2021-12-31"   # train window end when val is held out
VAL_START        = "2022-01-01"   # validation window start
TRAIN_END        = "2022-12-31"   # end of val window (= full train+val range)
TEST_START       = "2023-01-01"
TEST_END         = "2025-01-31"
