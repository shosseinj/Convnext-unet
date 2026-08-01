import tempfile
import unittest
from pathlib import Path

from tools.finalize_controls import require_all_complete
from tools.run_control_queue import load_control_jobs


class FinalizeControlTests(unittest.TestCase):
    def test_control_finalization_refuses_incomplete_matrix(self):
        jobs = load_control_jobs("configs/ablation_matrix.yaml")
        self.assertEqual(len(jobs), 42)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "42 run\(s\) incomplete"):
                require_all_complete(Path(directory), jobs)


if __name__ == "__main__":
    unittest.main()
