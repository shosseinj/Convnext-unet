import tempfile
import unittest
from pathlib import Path

import yaml

from tools.run_control_queue import load_control_jobs, require_incremental_gate


class ControlQueueTests(unittest.TestCase):
    def test_jobs_come_from_independent_controls_only(self):
        jobs = load_control_jobs("configs/ablation_matrix.yaml")
        self.assertEqual(len(jobs), 42)
        self.assertEqual(len(set(jobs)), 42)
        self.assertNotIn(("control_gdf_concatenation", 42), jobs)

    def test_incremental_gate_refuses_missing_runs(self):
        matrix = yaml.safe_load(Path("configs/ablation_matrix.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "18-run incremental gate"):
                require_incremental_gate(Path(directory), matrix)


if __name__ == "__main__":
    unittest.main()
