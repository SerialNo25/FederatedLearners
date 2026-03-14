import unittest

from stages.local_training.config import LocalTrainingConfig


class LocalTrainingConfigTests(unittest.TestCase):
    def test_accepts_federated_style_payload_and_selects_requested_institution(self):
        config = LocalTrainingConfig.from_dict(
            {
                "experiment_name": "local_exp",
                "output_dir": "data/experiments",
                "num_institutions": 2,
                "num_rounds": 3,
                "local_epochs": 2,
                "learning_rate": 0.01,
                "proximal_mu": 0.1,
                "local_institution_id": "bank_2",
                "model": {"model_type": "logistic_regression"},
                "institutions": [
                    {
                        "institution_id": "bank_1",
                        "dataset_path": "configs/sample_data/bank_1.csv",
                    },
                    {
                        "institution_id": "bank_2",
                        "dataset_path": "configs/sample_data/bank_2.csv",
                    },
                ],
            }
        )

        self.assertEqual(config.selected_institution.institution_id, "bank_2")

    def test_defaults_to_first_institution_when_local_id_missing(self):
        config = LocalTrainingConfig.from_dict(
            {
                "local_epochs": 1,
                "learning_rate": 0.01,
                "model": {"model_type": "logistic_regression"},
                "institutions": [
                    {
                        "institution_id": "bank_1",
                        "dataset_path": "configs/sample_data/bank_1.csv",
                    },
                    {
                        "institution_id": "bank_2",
                        "dataset_path": "configs/sample_data/bank_2.csv",
                    },
                ],
            }
        )

        self.assertEqual(config.selected_institution.institution_id, "bank_1")


if __name__ == "__main__":
    unittest.main()
