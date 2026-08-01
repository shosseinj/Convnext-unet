import tempfile
import unittest
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.generate_incremental_result_artifacts import DATASETS, VARIANTS, generate, load_aggregates


class IncrementalResultArtifactTests(unittest.TestCase):
    def test_missing_aggregate_blocks_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing validated aggregate"):
                load_aggregates(Path(directory))

    def test_generation_from_validated_in_memory_aggregates(self):
        stats = {"mean": 0.8, "sample_std": 0.01, "ci95_lower": 0.77, "ci95_upper": 0.83,
                 "values": {"42": 0.79, "3407": 0.8, "2026": 0.81}}
        loaded = {}
        for variant in VARIANTS:
            loaded[variant] = {"datasets": {}}
            for dataset in DATASETS:
                loaded[variant]["datasets"][dataset] = {
                    "metrics": {"dice": dict(stats), "iou": dict(stats)},
                    "size_stratified": {
                        label: {"dice": dict(stats), "iou": dict(stats)}
                        for label in ("small", "medium", "large")
                    },
                }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = generate(loaded, root)
            self.assertEqual(result, {"rows": 30, "size_rows": 30})
            with (root / "tables" / "incremental_results.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 30)
            ET.parse(root / "figures" / "incremental_results.svg")
            self.assertIn("descriptive observations",
                          (root / "sections" / "incremental_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
