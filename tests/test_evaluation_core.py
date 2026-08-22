import os
from pathlib import Path
import subprocess
import sys
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

    def test_clinicdb_disguised_tiff_decodes_without_libtiff_warning_spam(self):
        root = Path(__file__).resolve().parents[1]
        image = next((root / "data" / "CVC-ClinicDB" / "images").glob("*.png"))
        environment = dict(os.environ)
        environment.pop("OPENCV_LOG_LEVEL", None)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import evaluation_core, cv2; "
                f"image=cv2.imread({str(image)!r}, cv2.IMREAD_COLOR); "
                "assert image is not None; print(image.shape)",
            ],
            cwd=root, env=environment, capture_output=True, text=True, check=True,
        )
        self.assertNotIn("TIFF_Warning", completed.stderr)


if __name__ == "__main__":
    unittest.main()
