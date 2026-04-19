import unittest

from domain.training.class_weighting import compute_binary_class_balance


class ClassWeightingTests(unittest.TestCase):
    def test_computes_positive_weight_from_negative_positive_ratio(self):
        balance = compute_binary_class_balance([0, 0, 0, 1])

        self.assertEqual(balance.negatives, 3)
        self.assertEqual(balance.positives, 1)
        self.assertEqual(balance.pos_weight, 3.0)
        self.assertEqual(balance.positive_rate, 0.25)

    def test_rejects_missing_positive_class(self):
        with self.assertRaises(ValueError):
            compute_binary_class_balance([0, 0, 0])

    def test_rejects_missing_negative_class(self):
        with self.assertRaises(ValueError):
            compute_binary_class_balance([1, 1, 1])


if __name__ == "__main__":
    unittest.main()
