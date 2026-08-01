import unittest

import torch

from research_pipeline.evaluation import SizeBins, aggregate_records, binary_metrics


class OfficialEvaluationTests(unittest.TestCase):
    def test_binary_metrics_and_empty_semantics(self):
        probability = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]],
                                    [[[0.1, 0.1], [0.1, 0.1]]]])
        target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]],
                               [[[0.0, 0.0], [0.0, 0.0]]]])
        rows = binary_metrics(probability, target)
        self.assertAlmostEqual(rows[0]["dice"], 2 / 3)
        self.assertAlmostEqual(rows[0]["iou"], 0.5)
        self.assertAlmostEqual(rows[0]["precision"], 0.5)
        self.assertEqual(rows[0]["recall"], 1.0)
        self.assertEqual(rows[1]["dice"], 1.0)
        self.assertEqual(rows[1]["precision"], 1.0)
        self.assertEqual(rows[1]["recall"], 1.0)

    def test_shape_validation_size_bins_and_aggregation(self):
        with self.assertRaisesRegex(ValueError, "identical NCHW"):
            binary_metrics(torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2))
        bins = SizeBins()
        self.assertEqual([bins.label(value) for value in (0.01, 0.05, 0.20, 0.21)],
                         ["small", "medium", "medium", "large"])
        summary = aggregate_records([{"dice": 0.5}, {"dice": 1.0}], ("dice",))
        self.assertEqual(summary, {"dice": 0.75, "samples": 2})


if __name__ == "__main__":
    unittest.main()
