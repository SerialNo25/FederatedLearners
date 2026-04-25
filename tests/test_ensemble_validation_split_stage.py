import json
import tempfile
import unittest
from pathlib import Path

from domain.dataset.dataset_loader import load_institution_dataset, split_for_local_training
from domain.dataset.schema import ALL_COLUMNS
from stages.ensemble_validation_split.config import EnsembleValidationSplitConfig
from stages.ensemble_validation_split.stage import EnsembleValidationSplitStage


def _write_dataset(path: Path, labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(ALL_COLUMNS) + "\n")
        for index, label in enumerate(labels):
            features = [str(float(index * 100 + feature_index)) for feature_index in range(len(ALL_COLUMNS) - 1)]
            handle.write(",".join([*features, str(label)]) + "\n")


class EnsembleValidationSplitStageTests(unittest.TestCase):
    def test_writes_exact_local_training_validation_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "bank_1_train.csv"
            _write_dataset(dataset_path, labels=[0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0])

            config = EnsembleValidationSplitConfig.from_dict(
                {
                    "output_dir": str(root / "ensemble_validation"),
                    "seed": 17,
                    "institutions": [
                        {
                            "institution_id": "bank_1",
                            "dataset_path": str(dataset_path),
                        }
                    ],
                }
            )

            result = EnsembleValidationSplitStage(config).execute()

            output_path = result / "bank_1_validation.csv"
            metadata_path = result / "bank_1_validation.json"
            self.assertEqual(result, root / "ensemble_validation")
            self.assertTrue(output_path.exists())
            self.assertTrue(metadata_path.exists())

            source_dataset = load_institution_dataset("bank_1", dataset_path)
            _, expected_validation = split_for_local_training(source_dataset, seed=17)
            written_validation = load_institution_dataset("bank_1", output_path)

            self.assertEqual(written_validation.features, expected_validation.features)
            self.assertEqual(written_validation.labels, expected_validation.labels)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["matches_local_training_split"])
            self.assertEqual(metadata["rows_validation"], len(expected_validation.labels))


if __name__ == "__main__":
    unittest.main()
