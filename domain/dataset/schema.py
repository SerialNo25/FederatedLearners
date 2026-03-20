"""Shared dataset schema definitions for fraud detection experiments."""

from __future__ import annotations

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "hour_of_day",
    "day_of_week"
]
TARGET_COLUMN = "is_fraud"
ALL_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
