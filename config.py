"""Shared configuration for backtesting notebooks."""

# Date range
S_DATE = "2015-07-01"  # earliest valid date (Wikipedia pageviews start here)

# Columns to exclude from model features
EXCLUDE_VARS = ("Open", "High", "Low", "Close", "Volume")

# Walk-forward prediction start points
# Uncomment the one you want to use in notebooks
# INITIAL_TRAIN_PERIOD = 1890  # 2015-07-01 → start predicting in 2023
INITIAL_TRAIN_PERIOD = 2140    # 2015-07-01 → start predicting in 2024
# INITIAL_TRAIN_PERIOD = 2390  # 2015-07-01 → start predicting in 2025
# INITIAL_TRAIN_PERIOD = 7535  # 1993-01-29 → start predicting in 2024

# Transaction cost as a fraction of trade value (e.g. 0.001 = 0.1%)
TRANSACTION_COST = 0.001

# Reproducibility
RANDOM_STATE = 42
