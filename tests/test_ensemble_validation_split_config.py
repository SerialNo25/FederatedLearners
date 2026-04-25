import unittest

from stages.ensemble_validation_split.config import EnsembleValidationSplitConfig


def _base_dict(**overrides) -> dict:
    return {
        "institutions": [
            {
                "institution_id": "bank_1",
                "dataset_path": "data/train_test_splits/bank_1_train.csv",
            }
        ],
        **overrides,
    }


class EnsembleValidationSplitConfigTests(unittest.TestCase):
    def test_defaults_applied(self):
        config = EnsembleValidationSplitConfig.from_dict(_base_dict())
        self.assertEqual(config.stage, "ensemble_validation_split")
        self.assertEqual(config.validation_fraction, 0.2)
        self.assertEqual(config.seed, 42)

    def test_invalid_validation_fraction_rejected(self):
        with self.assertRaises(ValueError):
            EnsembleValidationSplitConfig.from_dict(_base_dict(validation_fraction=1.0))

    def test_duplicate_institutions_rejected(self):
        with self.assertRaises(ValueError):
            EnsembleValidationSplitConfig.from_dict(
                _base_dict(
                    institutions=[
                        {
                            "institution_id": "bank_1",
                            "dataset_path": "data/train_test_splits/bank_1_train.csv",
                        },
                        {
                            "institution_id": "bank_1",
                            "dataset_path": "data/train_test_splits/bank_1_train_copy.csv",
                        },
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
