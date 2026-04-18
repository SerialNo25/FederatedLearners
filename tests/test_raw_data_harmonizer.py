import csv
import json
import tempfile
import unittest
from pathlib import Path

from domain.harmonization.raw_data_harmonizer import (
    RawDataHarmonizationService,
    RawDatasetSource,
)


class RawDataHarmonizerTests(unittest.TestCase):
    def test_split_harmonization_fits_stats_on_train_subset_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            raw_path = workdir / "banksim.csv"
            self._write_banksim_fixture(raw_path)

            summary = RawDataHarmonizationService(seed=7).harmonize_train_test_split(
                source=RawDatasetSource(
                    institution_id="bank_2",
                    bank_kind="banksim",
                    raw_path=raw_path,
                    output_filename="unused.csv",
                ),
                output_dir=workdir / "out",
                test_fraction=0.5,
            )

            train_amounts = self._read_amounts(summary.train.output_path)
            test_amounts = self._read_amounts(summary.test.output_path)
            artifact = json.loads(summary.preprocessing_artifact_path.read_text(encoding="utf-8"))

            self.assertEqual(summary.train.row_count, 2)
            self.assertEqual(summary.test.row_count, 2)
            self.assertEqual(len(train_amounts), 2)
            self.assertEqual(len(test_amounts), 2)
            self.assertAlmostEqual(
                artifact["statistics"]["amount_mean"],
                sum(train_amounts) / len(train_amounts),
            )
            self.assertNotEqual(
                artifact["statistics"]["amount_mean"],
                (sum(train_amounts) + sum(test_amounts)) / (len(train_amounts) + len(test_amounts)),
            )
            self.assertEqual(artifact["fit_subset"], "train")

    def _write_banksim_fixture(self, path: Path) -> None:
        rows = [
            {"step": "0", "amount": "10", "age": "2", "zipcodeOri": "es_1", "zipMerchant": "es_2", "gender": "M", "category": "es_food", "fraud": "0"},
            {"step": "1", "amount": "20", "age": "3", "zipcodeOri": "es_1", "zipMerchant": "es_3", "gender": "F", "category": "es_fashion", "fraud": "0"},
            {"step": "2", "amount": "100", "age": "4", "zipcodeOri": "es_1", "zipMerchant": "es_4", "gender": "M", "category": "es_health", "fraud": "1"},
            {"step": "3", "amount": "200", "age": "5", "zipcodeOri": "es_1", "zipMerchant": "es_5", "gender": "F", "category": "es_travel", "fraud": "1"},
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _read_amounts(self, path: Path) -> list[float]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [float(row["amount"]) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    unittest.main()
