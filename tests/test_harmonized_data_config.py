import unittest

from stages.harmonized_data.config import HarmonizedDataConfig


def _base_dict(**overrides) -> dict:
    return {
        "output_dir": "data/harmonized",
        "seed": 42,
        "sparkov_target_size": 500000,
        "datasets": [
            {
                "institution_id": "bank_1",
                "bank_kind": "sparkov",
                "raw_path": "data/raw/Bank_A.csv",
                "output_filename": "bank_a_sparkov.csv",
            },
            {
                "institution_id": "bank_2",
                "bank_kind": "banksim",
                "raw_path": "data/raw/Bank_B.csv",
                "output_filename": "bank_b_banksim.csv",
            },
        ],
        **overrides,
    }


class HarmonizedDataConfigTests(unittest.TestCase):
    def test_defaults_load(self):
        config = HarmonizedDataConfig.from_dict(_base_dict())
        self.assertEqual(config.output_dir.as_posix(), "data/harmonized")
        self.assertEqual(config.sparkov_target_size, 500000)

    def test_requires_unique_institution_ids(self):
        with self.assertRaises(ValueError):
            HarmonizedDataConfig.from_dict(
                _base_dict(
                    datasets=[
                        {
                            "institution_id": "bank_1",
                            "bank_kind": "sparkov",
                            "raw_path": "data/raw/Bank_A.csv",
                            "output_filename": "bank_a_sparkov.csv",
                        },
                        {
                            "institution_id": "bank_1",
                            "bank_kind": "banksim",
                            "raw_path": "data/raw/Bank_B.csv",
                            "output_filename": "bank_b_banksim.csv",
                        },
                    ]
                )
            )

    def test_requires_positive_sparkov_target_size(self):
        with self.assertRaises(ValueError):
            HarmonizedDataConfig.from_dict(_base_dict(sparkov_target_size=0))


if __name__ == "__main__":
    unittest.main()
