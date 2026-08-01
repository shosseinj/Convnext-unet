import unittest

import numpy as np
import torch

from video_evaluation import binary_metrics, filter_components, resize_probability


class VideoEvaluationTests(unittest.TestCase):
    def test_empty_masks_and_metrics(self):
        mask = np.zeros((10, 12), np.uint8)
        filtered, components = filter_components(mask, 1)
        self.assertFalse(filtered.any())
        self.assertEqual(components, [])
        scores = binary_metrics(mask, mask)
        self.assertEqual(scores["dice"], 1.0)
        self.assertEqual(scores["iou"], 1.0)

    def test_single_component_bbox(self):
        mask = np.zeros((20, 30), np.uint8)
        mask[3:8, 7:16] = 1
        _, components = filter_components(mask, 1)
        self.assertEqual(len(components), 1)
        self.assertEqual((components[0]["x"], components[0]["y"],
                          components[0]["width"], components[0]["height"]), (7, 3, 9, 5))

    def test_multiple_components(self):
        mask = np.zeros((30, 30), np.uint8)
        mask[1:5, 2:6] = 1
        mask[18:25, 20:28] = 1
        _, components = filter_components(mask, 1)
        self.assertEqual(len(components), 2)

    def test_removes_small_component(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[1:3, 1:3] = 1
        mask[10:15, 10:15] = 1
        filtered, components = filter_components(mask, 10)
        self.assertEqual(len(components), 1)
        self.assertEqual(int(filtered.sum()), 25)

    def test_probability_resize(self):
        probability = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        resized = resize_probability(probability, 8, 10)
        self.assertEqual(resized.shape, (8, 10))
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
