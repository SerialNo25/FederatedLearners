import unittest

from stages.dataset_mixer.config import DatasetMixerConfig


def _base_dict(**overrides) -> dict:
    return {
        "output_path": "data/mixed_datasets/mixed.csv",
        "mixed_institution_id": "mixed_dataset",
        "seed": 42,
        "institutions": [
            {
                "institution_id": "bank_1",
                "dataset_path": "data/train_test_splits/bank_1_train.csv",
                "sample_size": 8000,
            },
            {
                "institution_id": "bank_2",
                "dataset_path": "data/train_test_splits/bank_2_train.csv",
                "sample_size": 1000,
            },
            {
                "institution_id": "bank_3",
                "dataset_path": "data/train_test_splits/bank_3_train.csv",
                "sample_size": 1000,
            },
        ],
        **overrides,
    }


class DatasetMixerConfigTests(unittest.TestCase):
    def test_defaults_load(self):
        config = DatasetMixerConfig.from_dict(_base_dict())
        self.assertEqual(config.stage, "dataset_mixer")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.mixed_institution_id, "mixed_dataset")

    def test_requires_unique_institution_ids(self):
        with self.assertRaises(ValueError):
            DatasetMixerConfig.from_dict(
                _base_dict(
                    institutions=[
                        {
                            "institution_id": "bank_1",
                            "dataset_path": "data/train_test_splits/bank_1_train.csv",
                            "sample_size": 5000,
                        },
                        {
                            "institution_id": "bank_1",
                            "dataset_path": "data/train_test_splits/bank_2_train.csv",
                            "sample_size": 5000,
                        },
                    ]
                )
            )

    def test_requires_positive_sample_size(self):
        with self.assertRaises(ValueError):
            DatasetMixerConfig.from_dict(
                _base_dict(
                    institutions=[
                        {
                            "institution_id": "bank_1",
                            "dataset_path": "data/train_test_splits/bank_1_train.csv",
                            "sample_size": 0,
                        }
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
