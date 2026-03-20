import unittest

from domain.dataset.dataset_loader import InstitutionDataset, split_dataset


class DatasetSplitTests(unittest.TestCase):
    def test_single_example_class_stays_in_training_split(self):
        dataset = InstitutionDataset(
            institution_id="bank_1",
            features=[[0.0], [1.0], [2.0]],
            labels=[0, 0, 1],
        )

        train_dataset, val_dataset = split_dataset(dataset, val_fraction=0.5, seed=7)

        self.assertEqual(sum(train_dataset.labels), 1)
        self.assertEqual(sum(val_dataset.labels), 0)
        self.assertEqual(len(train_dataset.labels) + len(val_dataset.labels), len(dataset.labels))


if __name__ == "__main__":
    unittest.main()
