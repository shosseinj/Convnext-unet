import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "Run-45-FAFEM-Residual-FG-MSCB-Remaining-Seeds.ps1"


class Exp45RemainingSeedsRunnerTests(unittest.TestCase):
    def test_dry_run_resolves_only_6543_and_7777_with_exp45_flags(self):
        self.assertTrue(RUNNER.exists())
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        payload = json.loads(result.stdout.strip())
        resolved_seeds = []
        for item in payload:
            command = dict(zip(item["train"][::2], item["train"][1::2]))
            resolved_seeds.append(int(command["--seed"]))
            self.assertEqual(command["--experiment_name"], "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine")
            self.assertEqual(command["--enable_residual_fg_mscb_lite_stage3"], "True")
            self.assertEqual(command["--batch_size"], "24")
        self.assertEqual(resolved_seeds, [6543, 7777])

    def test_real_run_path_does_not_capture_child_console_output(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("if ($DryRun) { $dryResult = &", source)
        self.assertIn("else {\n            & $invokeSeed $seed $false", source)


if __name__ == "__main__":
    unittest.main()
