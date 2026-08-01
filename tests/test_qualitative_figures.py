import unittest

import numpy as np

from tools.generate_qualitative_figures import colorize_masks, deterministic_selection


class QualitativeFigureTests(unittest.TestCase):
    def test_selection_depends_on_area_and_filename_only(self):
        rows = [
            {"image": "b.png", "size_bin": "small", "mask_area_ratio": "0.01", "dice": "0.1"},
            {"image": "a.png", "size_bin": "small", "mask_area_ratio": "0.03", "dice": "0.9"},
            {"image": "c.png", "size_bin": "small", "mask_area_ratio": "0.02", "dice": "0.0"},
        ]
        chosen = deterministic_selection(rows)
        self.assertEqual(chosen[0]["image"], "c.png")

    def test_prediction_and_error_colors_are_explicit(self):
        truth = np.array([[1, 1], [0, 0]], dtype=bool)
        prediction = np.array([[1, 0], [1, 0]], dtype=bool)
        ground_truth, predicted, error = colorize_masks(truth, prediction)
        self.assertEqual(tuple(predicted[0, 0]), (236, 64, 180))
        self.assertEqual(tuple(error[0, 0]), (255, 255, 255))
        self.assertEqual(tuple(error[1, 0]), (255, 215, 0))
        self.assertEqual(tuple(error[0, 1]), (0, 206, 209))
        self.assertEqual(tuple(ground_truth[1, 1]), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
