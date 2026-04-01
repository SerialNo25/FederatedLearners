"""Shared dataset schema definitions for fraud detection experiments."""

from __future__ import annotations

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "amount_zscore",
    "amount_percentile",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_night",
    "is_round_amount",
    "gender_M",
    "age_normalized",
    "geo_encoded",
    "cat_grocery",
    "cat_shopping",
    "cat_entertainment",
    "cat_gas_transport",
    "cat_food_dining",
    "cat_health_personal",
    "cat_other",
]
TARGET_COLUMN = "is_fraud"
ALL_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
