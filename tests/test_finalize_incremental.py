import tempfile
import unittest
from pathlib import Path

from tools.finalize_incremental import identities, require_all_complete


class FinalizeIncrementalTests(unittest.TestCase):
    def test_canonical_identity_count(self):
        jobs = identities("configs/ablation_matrix.yaml")
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len(set(jobs)), 18)

    def test_finalization_refuses_incomplete_matrix(self):
        jobs = identities("configs/ablation_matrix.yaml")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "18 run\(s\) incomplete"):
                require_all_complete(Path(directory), jobs)


if __name__ == "__main__":
    unittest.main()
