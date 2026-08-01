import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from research_pipeline.reproducibility import make_split_manifest


class ReproducibilityTests(unittest.TestCase):
    def make_data(self, root):
        for dataset in ("Kvasir-SEG", "CVC-ClinicDB"):
            for folder in ("images", "masks"):
                (root / dataset / folder).mkdir(parents=True)
            for index in range(10):
                image = np.full((4, 4, 3), index, np.uint8)
                cv2.imwrite(str(root / dataset / "images" / f"{index}.png"), image)
                cv2.imwrite(str(root / dataset / "masks" / f"{index}.png"), image[:, :, 0])

    def test_same_seed_same_manifest_and_no_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.make_data(root)
            first = make_split_manifest(root, ["Kvasir-SEG", "CVC-ClinicDB"], 42, 0.2)
            second = make_split_manifest(root, ["Kvasir-SEG", "CVC-ClinicDB"], 42, 0.2)
            self.assertEqual(first, second)
            for split in first["datasets"].values():
                train = {item["image"] for item in split["train"]}
                validation = {item["image"] for item in split["validation"]}
                self.assertFalse(train & validation)
                self.assertEqual(len(train), 8)
                self.assertEqual(len(validation), 2)

    def test_different_seeds_change_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.make_data(root)
            first = make_split_manifest(root, ["Kvasir-SEG"], 42, 0.2)
            second = make_split_manifest(root, ["Kvasir-SEG"], 3407, 0.2)
            self.assertNotEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
