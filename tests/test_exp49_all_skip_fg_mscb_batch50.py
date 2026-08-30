import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "49_fafem_residual_frequency_guided_mscb_all_skips_batch32_seed42.ps1"


class Exp49Batch32Tests(unittest.TestCase):
    def test_runner_uses_isolated_experiment_and_batch_size_32(self):
        self.assertTrue(RUNNER.exists())
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        command = dict(zip(json.loads(result.stdout.strip())["train"][::2],
                           json.loads(result.stdout.strip())["train"][1::2]))
        self.assertEqual(command["--experiment_name"], "one_seed_49_fafem_residual_frequency_guided_mscb_all_skips_batch32")
        self.assertEqual(command["--batch_size"], "32")
        self.assertEqual(command["--enable_residual_fg_mscb_all_skips"], "True")


if __name__ == "__main__":
    unittest.main()
