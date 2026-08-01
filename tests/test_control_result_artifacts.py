import tempfile
import unittest
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.generate_control_result_artifacts import DATASETS, FAMILIES, generate, load_controls


class ControlResultArtifactTests(unittest.TestCase):
    def test_missing_aggregates_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing validated aggregate"):
                load_controls(Path(directory), "configs/ablation_matrix.yaml")

    def test_generation_preserves_provenance(self):
        stats = {"mean": 0.8, "sample_std": 0.01, "ci95_lower": 0.77, "ci95_upper": 0.83,
                 "values": {"42": 0.79, "3407": 0.8, "2026": 0.81}}
        loaded = {}
        for _, variants in FAMILIES:
            for variant in variants:
                loaded[variant] = {"source_variant": variant, "provenance": "independent_trained",
                                   "aggregate": {"datasets": {
                                       dataset: {"metrics": {"dice": dict(stats), "iou": dict(stats)}}
                                       for dataset in DATASETS}}}
        loaded["control_msc_branches_3"]["source_variant"] = "baseline_msc"
        loaded["control_msc_branches_3"]["provenance"] = "reused:baseline_msc"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = generate(loaded, root)
            self.assertEqual(result, {"rows": 125, "families": 7})
            with (root / "tables" / "control_results.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 125)
            reused = [row for row in rows if row["control"] == "control_msc_branches_3"]
            self.assertTrue(all(row["provenance"] == "reused:baseline_msc" for row in reused))
            ET.parse(root / "figures" / "control_results.svg")


if __name__ == "__main__":
    unittest.main()
