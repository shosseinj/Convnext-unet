import math
import json
import tempfile
import unittest
from pathlib import Path

from tools.aggregate_official import load_seed_summaries, summarize


class OfficialAggregationTests(unittest.TestCase):
    def test_three_seed_sample_statistics(self):
        result = summarize([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_std"], 1.0)
        expected = 4.302652729911275 / math.sqrt(3)
        self.assertAlmostEqual(result["ci95_lower"], 2.0 - expected)
        self.assertAlmostEqual(result["ci95_upper"], 2.0 + expected)
        self.assertEqual(result["values"], {"42": 1.0, "3407": 2.0, "2026": 3.0})

    def test_rejects_incomplete_or_nonfinite_seed_sets(self):
        with self.assertRaisesRegex(ValueError, "exactly three finite"):
            summarize([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "exactly three finite"):
            summarize([1.0, float("nan"), 3.0])

    def test_source_summaries_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 3407, 2026):
                path = root / "evaluation" / "v" / f"seed_{seed}" / "Kvasir-SEG" / "summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"status": "PASS", "variant": "v", "seed": seed,
                                            "dataset": "Kvasir-SEG", "threshold": 0.5, "tta": False}),
                                encoding="utf-8")
            rows = load_seed_summaries(root, "v", "Kvasir-SEG")
            self.assertTrue(all(len(row["_source_sha256"]) == 64 for row in rows))
            self.assertTrue(all(Path(row["_source_path"]).is_file() for row in rows))


if __name__ == "__main__":
    unittest.main()
