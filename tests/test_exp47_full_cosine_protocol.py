import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "47_fafem_residual_frequency_guided_deformable_mscb_stage3_full_cosine_seed42.ps1"


class Exp47FullCosineProtocolTests(unittest.TestCase):
    def test_runner_keeps_exp46_configuration_and_defers_early_stopping(self):
        self.assertTrue(RUNNER.exists())
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout.strip())
        command = dict(zip(payload["train"][::2], payload["train"][1::2]))
        self.assertEqual(command["--experiment_name"], "one_seed_47_fafem_residual_frequency_guided_deformable_mscb_stage3_full_cosine")
        self.assertEqual(command["--epochs"], "200")
        self.assertEqual(command["--lr_scheduler"], "warmup_cosine")
        self.assertEqual(command["--warmup_epochs"], "5")
        self.assertEqual(command["--early_stop_start_epoch"], "160")
        self.assertEqual(command["--early_stop_patience"], "30")
        self.assertEqual(command["--enable_deformable_residual_fg_mscb_lite_stage3"], "True")


if __name__ == "__main__":
    unittest.main()
