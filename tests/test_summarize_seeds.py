import csv
import json
import tempfile
import unittest
from pathlib import Path

from ablation_artifacts import DATASETS, METRICS
from summarize_seeds import summarize


class SummarizeSeedsTests(unittest.TestCase):
    def test_mean_sample_std_and_single_complexity_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            for seed, value in zip((42, 6543, 7777), (0.7, 0.8, 0.9)):
                seed_dir = experiment_dir / f"seed_{seed}"
                seed_dir.mkdir()
                payload = {
                    "experiment_name": "01_baseline", "seed": seed,
                    "trainable_parameters": 10, "total_parameters": 20,
                    "macs": 100, "flops": 200, "gmacs": 1e-7, "gflops": 2e-7,
                    "results": {
                        dataset: {metric: value for metric in METRICS}
                        for dataset in DATASETS
                    },
                }
                (seed_dir / "evaluation_summary.json").write_text(json.dumps(payload))
            output = summarize(experiment_dir, "01_baseline", validate_fingerprint=False)
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            metric_rows = [row for row in rows if row["row_type"] == "metric"]
            complexity_rows = [row for row in rows if row["row_type"] == "complexity"]
            self.assertEqual(len(metric_rows), 35)
            self.assertEqual(len(complexity_rows), 6)
            self.assertAlmostEqual(float(metric_rows[0]["mean"]), 0.8)
            self.assertAlmostEqual(float(metric_rows[0]["sample_std"]), 0.1)


if __name__ == "__main__":
    unittest.main()
