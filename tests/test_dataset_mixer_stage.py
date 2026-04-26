import json
import tempfile
import unittest
from pathlib import Path

from domain.dataset.schema import ALL_COLUMNS
from stages.dataset_mixer.config import DatasetMixerConfig
from stages.dataset_mixer.stage import DatasetMixerStage


def _write_dataset(path: Path, row_count: int, fraud_every: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(ALL_COLUMNS) + "\n")
        for index in range(row_count):
            features = [str(float(index + feature_index)) for feature_index in range(len(ALL_COLUMNS) - 1)]
            label = "1" if index % fraud_every == 0 else "0"
            handle.write(",".join([*features, label]) + "\n")


class DatasetMixerStageTests(unittest.TestCase):
    def test_samples_and_writes_mixed_dataset_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bank_1 = root / "bank_1.csv"
            bank_2 = root / "bank_2.csv"
            bank_3 = root / "bank_3.csv"
            _write_dataset(bank_1, row_count=20, fraud_every=2)
            _write_dataset(bank_2, row_count=20, fraud_every=3)
            _write_dataset(bank_3, row_count=20, fraud_every=4)

            output_path = root / "mixed" / "dominant_bank_1.csv"
            config = DatasetMixerConfig.from_dict(
                {
                    "output_path": str(output_path),
                    "mixed_institution_id": "mixed_bank_1",
                    "seed": 7,
                    "institutions": [
                        {
                            "institution_id": "bank_1",
                            "dataset_path": str(bank_1),
                            "sample_size": 8,
                        },
                        {
                            "institution_id": "bank_2",
                            "dataset_path": str(bank_2),
                            "sample_size": 1,
                        },
                        {
                            "institution_id": "bank_3",
                            "dataset_path": str(bank_3),
                            "sample_size": 1,
                        },
                    ],
                }
            )

            result = DatasetMixerStage(config).execute()

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.with_suffix(".json").exists())

            rows = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 11)
            self.assertEqual(rows[0], ",".join(ALL_COLUMNS))

            metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["mixed_institution_id"], "mixed_bank_1")
            self.assertEqual(metadata["total_rows"], 10)
            self.assertEqual(
                [source["requested_rows"] for source in metadata["sources"]],
                [8, 1, 1],
            )

    def test_rejects_sampling_more_rows_than_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bank_1 = root / "bank_1.csv"
            _write_dataset(bank_1, row_count=5, fraud_every=2)

            config = DatasetMixerConfig.from_dict(
                {
                    "output_path": str(root / "mixed.csv"),
                    "institutions": [
                        {
                            "institution_id": "bank_1",
                            "dataset_path": str(bank_1),
                            "sample_size": 6,
                        }
                    ],
                }
            )

            with self.assertRaises(ValueError):
                DatasetMixerStage(config).execute()


if __name__ == "__main__":
    unittest.main()
