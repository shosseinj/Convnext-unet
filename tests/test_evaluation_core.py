import unittest

import torch

from evaluation_core import binary_metrics_per_image


class EvaluationCoreTests(unittest.TestCase):
    def test_binary_metrics_are_mean_per_image(self):
        prediction = torch.tensor([
            [[[1, 0], [0, 0]]],
            [[[1, 1], [0, 0]]],
        ], dtype=torch.float32)
        target = torch.tensor([
            [[[1, 0], [0, 0]]],
            [[[1, 0], [1, 0]]],
        ], dtype=torch.float32)
        dice, iou = binary_metrics_per_image(prediction, target)
        self.assertAlmostEqual(dice, (1.0 + 0.5) / 2.0, places=6)
        self.assertAlmostEqual(iou, (1.0 + (1.0 / 3.0)) / 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
