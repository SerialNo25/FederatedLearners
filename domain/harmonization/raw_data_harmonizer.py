"""Transform raw bank datasets into the shared harmonized schema."""

from __future__ import annotations

import csv
import json
import math
import random
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from domain.dataset.schema import ALL_COLUMNS, TARGET_COLUMN

BankKind = Literal["sparkov", "banksim", "ccfraud"]

UNIFIED_CATEGORIES = [
    "grocery",
    "shopping",
    "entertainment",
    "gas_transport",
    "food_dining",
    "health_personal",
    "other",
]

SPARKOV_CAT_MAP = {
    "grocery_pos": "grocery",
    "grocery_net": "grocery",
    "shopping_pos": "shopping",
    "shopping_net": "shopping",
    "misc_pos": "shopping",
    "misc_net": "shopping",
    "entertainment": "entertainment",
    "travel": "entertainment",
    "gas_transport": "gas_transport",
    "food_dining": "food_dining",
    "health_fitness": "health_personal",
    "personal_care": "health_personal",
    "home": "other",
    "kids_pets": "other",
}

BANKSIM_CAT_MAP = {
    "es_food": "grocery",
    "es_hyper": "grocery",
    "es_sportsandtoys": "shopping",
    "es_fashion": "shopping",
    "es_tech": "shopping",
    "es_hotelservices": "entertainment",
    "es_leisure": "entertainment",
    "es_travel": "entertainment",
    "es_transportation": "gas_transport",
    "es_barsandrestaurants": "food_dining",
    "es_health": "health_personal",
    "es_wellnessandbeauty": "health_personal",
    "es_otherservices": "other",
    "es_contents": "other",
    "es_home": "other",
}

CCFRAUD_CAT_MAP = {
    "grocery": "grocery",
    "food": "grocery",
    "supermarket": "grocery",
    "shopping": "shopping",
    "retail": "shopping",
    "electronics": "shopping",
    "clothing": "shopping",
    "fashion": "shopping",
    "entertainment": "entertainment",
    "travel": "entertainment",
    "hotel": "entertainment",
    "leisure": "entertainment",
    "gas": "gas_transport",
    "transport": "gas_transport",
    "transportation": "gas_transport",
    "restaurant": "food_dining",
    "dining": "food_dining",
    "bar": "food_dining",
    "health": "health_personal",
    "beauty": "health_personal",
    "personal": "health_personal",
}

AGE_BRACKET_MAP = {
    "0": 25.0,
    "1": 30.0,
    "2": 35.0,
    "3": 40.0,
    "4": 50.0,
    "5": 60.0,
    "6": 70.0,
    "u": 40.0,
}


@dataclass(frozen=True)
class RawDatasetSource:
    institution_id: str
    bank_kind: BankKind
    raw_path: Path
    output_filename: str


@dataclass(frozen=True)
class HarmonizedDatasetSummary:
    institution_id: str
    bank_kind: BankKind
    output_path: Path
    row_count: int
    fraud_count: int


@dataclass(frozen=True)
class HarmonizedSubsetSummary:
    name: Literal["train", "test"]
    output_path: Path
    row_count: int
    fraud_count: int


@dataclass(frozen=True)
class HarmonizedSplitSummary:
    institution_id: str
    bank_kind: BankKind
    train: HarmonizedSubsetSummary
    test: HarmonizedSubsetSummary
    preprocessing_artifact_path: Path


@dataclass(frozen=True)
class _PrecomputedStats:
    amount_mean: float
    amount_std: float
    amount_percentiles: dict[float, float]
    amount_sorted_values: tuple[float, ...]
    geo_scale: float
    country_mapping: dict[str, int]
    selected_row_indices: set[int] | None


@dataclass(frozen=True)
class _RawRowStats:
    row_index: int
    label: int
    amount: float
    geo: float
    country: str


class RawDataHarmonizationService:
    def __init__(self, seed: int = 42, sparkov_target_size: int = 500_000) -> None:
        self._seed = seed
        self._sparkov_target_size = sparkov_target_size

    def harmonize(self, source: RawDatasetSource, output_dir: Path) -> HarmonizedDatasetSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = self._collect_stats(source)
        output_path = output_dir / source.output_filename
        row_count, fraud_count = self._write_harmonized_dataset(source, stats, output_path)
        return HarmonizedDatasetSummary(
            institution_id=source.institution_id,
            bank_kind=source.bank_kind,
            output_path=output_path,
            row_count=row_count,
            fraud_count=fraud_count,
        )

    def harmonize_train_test_split(
        self,
        source: RawDatasetSource,
        output_dir: Path,
        test_fraction: float,
    ) -> HarmonizedSplitSummary:
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = self._collect_raw_row_stats(source)
        selected_indices = self._select_source_indices(source, rows)
        train_indices, test_indices = self._split_indices_by_label(
            rows=rows,
            selected_indices=selected_indices,
            test_fraction=test_fraction,
        )
        train_stats = self._build_stats_from_rows(source, rows, train_indices)

        train_path = output_dir / f"{source.institution_id}_train.csv"
        test_path = output_dir / f"{source.institution_id}_test.csv"
        train_counts = self._write_harmonized_subset(
            source,
            train_stats,
            train_indices,
            train_path,
        )
        test_counts = self._write_harmonized_subset(
            source,
            train_stats,
            test_indices,
            test_path,
        )

        artifact_path = output_dir / f"{source.institution_id}_preprocessing.json"
        self._write_preprocessing_artifact(
            artifact_path=artifact_path,
            source=source,
            train_indices=train_indices,
            test_indices=test_indices,
            stats=train_stats,
            test_fraction=test_fraction,
        )

        return HarmonizedSplitSummary(
            institution_id=source.institution_id,
            bank_kind=source.bank_kind,
            train=HarmonizedSubsetSummary(
                name="train",
                output_path=train_path,
                row_count=train_counts[0],
                fraud_count=train_counts[1],
            ),
            test=HarmonizedSubsetSummary(
                name="test",
                output_path=test_path,
                row_count=test_counts[0],
                fraud_count=test_counts[1],
            ),
            preprocessing_artifact_path=artifact_path,
        )

    def _collect_stats(self, source: RawDatasetSource) -> _PrecomputedStats:
        amounts: list[float] = []
        geos: list[float] = []
        fraud_indices: list[int] = []
        legit_indices: list[int] = []
        country_values: set[str] = set()

        with source.raw_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                if self._row_has_missing_entries(row):
                    continue
                amount = self._extract_amount(source.bank_kind, row)
                geo = self._extract_geo_value(source.bank_kind, row)

                amounts.append(amount)
                geos.append(geo)

                if source.bank_kind == "sparkov":
                    label = self._extract_label(source.bank_kind, row)
                    if label == 1:
                        fraud_indices.append(row_index)
                    else:
                        legit_indices.append(row_index)

                if source.bank_kind == "ccfraud":
                    country_values.add(self._clean_string(row.get("Country of Transaction", "")))

        selected_row_indices: set[int] | None = None
        if source.bank_kind == "sparkov":
            selected_row_indices = self._build_sparkov_sample_indices(fraud_indices, legit_indices)

        amount_mean = float(np.mean(amounts)) if amounts else 0.0
        amount_std = float(np.std(amounts, ddof=1)) if len(amounts) > 1 else 0.0
        geo_scale = self._quantile_99(geos)
        country_mapping = {
            value: index
            for index, value in enumerate(sorted(country for country in country_values if country))
        }

        return _PrecomputedStats(
            amount_mean=amount_mean,
            amount_std=amount_std,
            amount_percentiles=self._build_percentile_lookup(amounts),
            amount_sorted_values=tuple(sorted(amounts)),
            geo_scale=geo_scale,
            country_mapping=country_mapping,
            selected_row_indices=selected_row_indices,
        )

    def _collect_raw_row_stats(self, source: RawDatasetSource) -> list[_RawRowStats]:
        rows: list[_RawRowStats] = []
        with source.raw_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                if self._row_has_missing_entries(row):
                    continue
                country = ""
                if source.bank_kind == "ccfraud":
                    country = self._clean_string(row.get("Country of Transaction", ""))
                rows.append(
                    _RawRowStats(
                        row_index=row_index,
                        label=self._extract_label(source.bank_kind, row),
                        amount=self._extract_amount(source.bank_kind, row),
                        geo=self._extract_geo_value(source.bank_kind, row),
                        country=country,
                    )
                )
        return rows

    def _select_source_indices(
        self,
        source: RawDatasetSource,
        rows: list[_RawRowStats],
    ) -> set[int]:
        if source.bank_kind != "sparkov":
            return {row.row_index for row in rows}

        fraud_indices = [row.row_index for row in rows if row.label == 1]
        legit_indices = [row.row_index for row in rows if row.label == 0]
        return self._build_sparkov_sample_indices(fraud_indices, legit_indices)

    def _split_indices_by_label(
        self,
        rows: list[_RawRowStats],
        selected_indices: set[int],
        test_fraction: float,
    ) -> tuple[set[int], set[int]]:
        indices_by_class: dict[int, list[int]] = {}
        for row in rows:
            if row.row_index in selected_indices:
                indices_by_class.setdefault(row.label, []).append(row.row_index)

        rng = random.Random(self._seed)
        train_indices: set[int] = set()
        test_indices: set[int] = set()
        for indices in indices_by_class.values():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            n_test = round(len(shuffled) * test_fraction)
            if len(shuffled) > 1:
                n_test = min(max(1, n_test), len(shuffled) - 1)
            else:
                n_test = 0
            test_indices.update(shuffled[:n_test])
            train_indices.update(shuffled[n_test:])

        return train_indices, test_indices

    def _build_stats_from_rows(
        self,
        source: RawDatasetSource,
        rows: list[_RawRowStats],
        selected_indices: set[int],
    ) -> _PrecomputedStats:
        selected_rows = [row for row in rows if row.row_index in selected_indices]
        amounts = [row.amount for row in selected_rows]
        geos = [row.geo for row in selected_rows]
        country_values = {row.country for row in selected_rows if row.country}

        amount_mean = float(np.mean(amounts)) if amounts else 0.0
        amount_std = float(np.std(amounts, ddof=1)) if len(amounts) > 1 else 0.0
        country_mapping = {
            value: index
            for index, value in enumerate(sorted(country_values))
        } if source.bank_kind == "ccfraud" else {}

        return _PrecomputedStats(
            amount_mean=amount_mean,
            amount_std=amount_std,
            amount_percentiles=self._build_percentile_lookup(amounts),
            amount_sorted_values=tuple(sorted(amounts)),
            geo_scale=self._quantile_99(geos),
            country_mapping=country_mapping,
            selected_row_indices=None,
        )

    def _write_harmonized_dataset(
        self,
        source: RawDatasetSource,
        stats: _PrecomputedStats,
        output_path: Path,
    ) -> tuple[int, int]:
        row_count = 0
        fraud_count = 0

        with source.raw_path.open("r", newline="", encoding="utf-8-sig") as input_handle:
            reader = csv.DictReader(input_handle)
            with output_path.open("w", newline="", encoding="utf-8") as output_handle:
                writer = csv.DictWriter(output_handle, fieldnames=ALL_COLUMNS)
                writer.writeheader()

                for row_index, row in enumerate(reader):
                    if self._row_has_missing_entries(row):
                        continue
                    if (
                        stats.selected_row_indices is not None
                        and row_index not in stats.selected_row_indices
                    ):
                        continue

                    harmonized = self._harmonize_row(source.bank_kind, row, stats)
                    writer.writerow(harmonized)
                    row_count += 1
                    fraud_count += int(harmonized[TARGET_COLUMN])

        return row_count, fraud_count

    def _write_harmonized_subset(
        self,
        source: RawDatasetSource,
        stats: _PrecomputedStats,
        selected_indices: set[int],
        output_path: Path,
    ) -> tuple[int, int]:
        row_count = 0
        fraud_count = 0

        with source.raw_path.open("r", newline="", encoding="utf-8-sig") as input_handle:
            reader = csv.DictReader(input_handle)
            with output_path.open("w", newline="", encoding="utf-8") as output_handle:
                writer = csv.DictWriter(output_handle, fieldnames=ALL_COLUMNS)
                writer.writeheader()

                for row_index, row in enumerate(reader):
                    if self._row_has_missing_entries(row):
                        continue
                    if row_index not in selected_indices:
                        continue

                    harmonized = self._harmonize_row(source.bank_kind, row, stats)
                    writer.writerow(harmonized)
                    row_count += 1
                    fraud_count += int(harmonized[TARGET_COLUMN])

        return row_count, fraud_count

    def _write_preprocessing_artifact(
        self,
        artifact_path: Path,
        source: RawDatasetSource,
        train_indices: set[int],
        test_indices: set[int],
        stats: _PrecomputedStats,
        test_fraction: float,
    ) -> None:
        payload = {
            "institution_id": source.institution_id,
            "bank_kind": source.bank_kind,
            "raw_path": str(source.raw_path),
            "seed": self._seed,
            "test_fraction": test_fraction,
            "split_policy": "stratified_by_raw_label_before_harmonization",
            "fit_subset": "train",
            "train_row_count": len(train_indices),
            "test_row_count": len(test_indices),
            "feature_columns": ALL_COLUMNS[:-1],
            "target_column": TARGET_COLUMN,
            "transforms": {
                "amount": "raw_amount",
                "log_amount": "log1p(max(raw_amount, 0.0))",
                "amount_zscore": "(raw_amount - train_amount_mean) / train_amount_std",
                "amount_percentile": "empirical_percentile_lookup_fit_on_train_amounts",
                "hour_sin": "sin(2*pi*hour/24)",
                "hour_cos": "cos(2*pi*hour/24)",
                "dow_sin": "sin(2*pi*day_of_week/7)",
                "dow_cos": "cos(2*pi*day_of_week/7)",
                "age_normalized": "clip((age_years - 18) / (100 - 18), 0, 1)",
                "geo_encoded": "raw_geo / train_geo_99th_percentile clipped to [0, 1]",
            },
            "statistics": {
                "amount_mean": stats.amount_mean,
                "amount_std": stats.amount_std,
                "amount_count": len(stats.amount_sorted_values),
                "amount_min": (
                    min(stats.amount_sorted_values) if stats.amount_sorted_values else None
                ),
                "amount_max": (
                    max(stats.amount_sorted_values) if stats.amount_sorted_values else None
                ),
                "geo_scale_99th_percentile": stats.geo_scale,
                "country_mapping": stats.country_mapping,
            },
        }
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _harmonize_row(
        self,
        bank_kind: BankKind,
        row: dict[str, str],
        stats: _PrecomputedStats,
    ) -> dict[str, float | int]:
        amount = self._extract_amount(bank_kind, row)
        hour_of_day, day_of_week = self._extract_time_parts(bank_kind, row)
        age_normalized = self._extract_age(bank_kind, row)
        geo_encoded = self._normalize_geo(bank_kind, row, stats)
        category = self._extract_category(bank_kind, row)
        label = self._extract_label(bank_kind, row)
        gender_m = self._extract_gender(bank_kind, row)

        harmonized: dict[str, float | int] = {
            "amount": amount,
            "log_amount": math.log1p(max(amount, 0.0)),
            "amount_zscore": self._amount_zscore(amount, stats),
            "amount_percentile": self._amount_percentile(amount, stats),
            "hour_sin": math.sin(2.0 * math.pi * hour_of_day / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * hour_of_day / 24.0),
            "dow_sin": math.sin(2.0 * math.pi * day_of_week / 7.0),
            "dow_cos": math.cos(2.0 * math.pi * day_of_week / 7.0),
            "is_weekend": int(day_of_week >= 5),
            "is_night": int(hour_of_day >= 22 or hour_of_day < 6),
            "is_round_amount": int(math.isclose(amount % 10.0, 0.0, abs_tol=1e-9)),
            "gender_M": gender_m,
            "age_normalized": age_normalized,
            "geo_encoded": geo_encoded,
            "cat_grocery": 0,
            "cat_shopping": 0,
            "cat_entertainment": 0,
            "cat_gas_transport": 0,
            "cat_food_dining": 0,
            "cat_health_personal": 0,
            "cat_other": 0,
            "is_fraud": label,
        }

        harmonized[f"cat_{category}"] = 1
        return {column: harmonized[column] for column in ALL_COLUMNS}

    def _amount_zscore(self, amount: float, stats: _PrecomputedStats) -> float:
        if stats.amount_std <= 1e-12:
            return 0.0
        return (amount - stats.amount_mean) / stats.amount_std

    def _amount_percentile(self, amount: float, stats: _PrecomputedStats) -> float:
        if amount in stats.amount_percentiles:
            return stats.amount_percentiles[amount]
        values = stats.amount_sorted_values
        if not values:
            return 0.5
        rank = bisect_right(values, amount)
        return self._clip01(rank / len(values))

    def _normalize_geo(
        self,
        bank_kind: BankKind,
        row: dict[str, str],
        stats: _PrecomputedStats,
    ) -> float:
        raw_geo = self._extract_geo_value(bank_kind, row)
        if bank_kind == "ccfraud":
            country = self._clean_string(row.get("Country of Transaction", ""))
            if not stats.country_mapping:
                return 0.5
            max_code = max(stats.country_mapping.values())
            code = stats.country_mapping.get(country)
            if code is None or max_code <= 0:
                return 0.0 if code == 0 else 0.5
            return self._clip01(code / max_code)

        scale = stats.geo_scale
        if scale <= 1e-12:
            return 0.0
        return self._clip01(raw_geo / scale)

    def _extract_time_parts(self, bank_kind: BankKind, row: dict[str, str]) -> tuple[int, int]:
        if bank_kind == "sparkov":
            timestamp = datetime.strptime(row["trans_date_trans_time"], "%Y-%m-%d %H:%M:%S")
            return timestamp.hour, timestamp.weekday()

        if bank_kind == "banksim":
            step = int(float(self._clean_string(row.get("step", "0")) or 0))
            return step % 24, (step // 24) % 7

        date_value = self._clean_string(row.get("Date", ""))
        time_value = self._clean_string(row.get("Time", ""))
        hour = self._parse_ccfraud_hour(time_value)

        if date_value:
            try:
                date_obj = datetime.strptime(date_value, "%d-%b-%y")
                return hour, date_obj.weekday()
            except ValueError:
                pass

        return hour, 3

    def _extract_age(self, bank_kind: BankKind, row: dict[str, str]) -> float:
        if bank_kind == "sparkov":
            dob_raw = self._clean_string(row.get("dob", ""))
            ts_raw = self._clean_string(row.get("trans_date_trans_time", ""))
            try:
                dob = datetime.strptime(dob_raw, "%Y-%m-%d")
                timestamp = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
                age_years = (timestamp - dob).days / 365.25
            except ValueError:
                age_years = 40.0
            return self._normalize_age(age_years)

        if bank_kind == "banksim":
            bracket = self._clean_string(row.get("age", "")).lower()
            return self._normalize_age(AGE_BRACKET_MAP.get(bracket, 40.0))

        age_value = self._safe_float(row.get("Age"), default=40.0)
        return self._normalize_age(age_value)

    def _extract_geo_value(self, bank_kind: BankKind, row: dict[str, str]) -> float:
        if bank_kind == "sparkov":
            lat = self._safe_float(row.get("lat"))
            lon = self._safe_float(row.get("long"))
            merch_lat = self._safe_float(row.get("merch_lat"))
            merch_lon = self._safe_float(row.get("merch_long"))
            return math.sqrt((lat - merch_lat) ** 2 + (lon - merch_lon) ** 2)

        if bank_kind == "banksim":
            zipcode_ori = self._safe_float(self._clean_string(row.get("zipcodeOri", "")).replace("es_", ""))
            zipcode_mer = self._safe_float(self._clean_string(row.get("zipMerchant", "")).replace("es_", ""))
            return abs(zipcode_ori - zipcode_mer)

        country = self._clean_string(row.get("Country of Transaction", ""))
        return 0.0 if not country else 1.0

    def _extract_amount(self, bank_kind: BankKind, row: dict[str, str]) -> float:
        if bank_kind == "sparkov":
            return self._safe_float(row.get("amt"))
        if bank_kind == "banksim":
            return self._safe_float(row.get("amount"))

        raw_amount = self._clean_string(row.get("Amount", ""))
        cleaned = "".join(ch for ch in raw_amount if ch.isdigit() or ch in ".-")
        return self._safe_float(cleaned)

    def _extract_gender(self, bank_kind: BankKind, row: dict[str, str]) -> int:
        column = "gender" if bank_kind != "ccfraud" else "Gender"
        value = self._clean_string(row.get(column, "")).upper()
        return int(value in {"M", "MALE"})

    def _extract_category(self, bank_kind: BankKind, row: dict[str, str]) -> str:
        if bank_kind == "sparkov":
            raw = row.get("category", "")
            mapping = SPARKOV_CAT_MAP
        elif bank_kind == "banksim":
            raw = row.get("category", "")
            mapping = BANKSIM_CAT_MAP
        else:
            raw = row.get("Merchant Group", "")
            mapping = CCFRAUD_CAT_MAP
        return self._map_category(self._clean_string(raw), mapping)

    def _extract_label(self, bank_kind: BankKind, row: dict[str, str]) -> int:
        column = {
            "sparkov": "is_fraud",
            "banksim": "fraud",
            "ccfraud": "Fraud",
        }[bank_kind]
        raw = self._clean_string(row.get(column, "")).lower()
        return int(raw in {"1", "true", "yes", "fraud"})

    def _build_sparkov_sample_indices(
        self,
        fraud_indices: list[int],
        legit_indices: list[int],
    ) -> set[int]:
        total_count = len(fraud_indices) + len(legit_indices)
        if total_count <= self._sparkov_target_size:
            return set(range(total_count))

        fraud_rate = len(fraud_indices) / total_count if total_count else 0.0
        target_fraud = min(len(fraud_indices), int(round(self._sparkov_target_size * fraud_rate)))
        target_legit = min(len(legit_indices), self._sparkov_target_size - target_fraud)

        if target_fraud + target_legit < self._sparkov_target_size:
            remaining = self._sparkov_target_size - (target_fraud + target_legit)
            extra_legit = min(len(legit_indices) - target_legit, remaining)
            target_legit += max(extra_legit, 0)

        rng = random.Random(self._seed)
        fraud_sample = rng.sample(fraud_indices, target_fraud) if len(fraud_indices) > target_fraud else fraud_indices
        legit_sample = rng.sample(legit_indices, target_legit) if len(legit_indices) > target_legit else legit_indices
        return set(fraud_sample) | set(legit_sample)

    def _build_percentile_lookup(self, amounts: Iterable[float]) -> dict[float, float]:
        values = list(amounts)
        if not values:
            return {}

        counts = Counter(values)
        percentiles: dict[float, float] = {}
        total = len(values)
        lower_rank = 0
        for amount in sorted(counts):
            count = counts[amount]
            average_rank = lower_rank + (count + 1) / 2.0
            percentiles[amount] = average_rank / total
            lower_rank += count
        return percentiles

    def _quantile_99(self, values: Iterable[float]) -> float:
        data = list(values)
        if not data:
            return 1.0
        return float(np.quantile(np.asarray(data, dtype=float), 0.99))

    def _map_category(self, value: str, mapping: dict[str, str]) -> str:
        normalized = value.strip().lower().replace(" ", "")
        if not normalized:
            return "other"
        if normalized in mapping:
            return mapping[normalized]
        for key, unified in mapping.items():
            if key in normalized or normalized in key:
                return unified
        return "other"

    def _parse_ccfraud_hour(self, raw: str) -> int:
        value = raw.strip()
        if not value:
            return 12

        try:
            return int(float(value)) % 24
        except ValueError:
            pass

        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(value, fmt).hour
            except ValueError:
                continue
        return 12

    def _normalize_age(self, age_years: float, min_age: float = 18.0, max_age: float = 100.0) -> float:
        return self._clip01((age_years - min_age) / (max_age - min_age))

    def _clip01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _row_has_missing_entries(self, row: dict[str, str | None]) -> bool:
        return any(self._clean_string(value or "") == "" for value in row.values())

    def _safe_float(self, value: str | None, default: float = 0.0) -> float:
        cleaned = self._clean_string(value or "")
        if not cleaned:
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default

    def _clean_string(self, value: str) -> str:
        return value.strip().strip("'").strip('"')
