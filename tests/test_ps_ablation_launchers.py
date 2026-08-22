import json
import subprocess
import unittest
from pathlib import Path

from ablation_cli import add_ablation_arguments
import argparse


ROOT = Path(__file__).resolve().parents[1]


class PowerShellAblationLauncherTests(unittest.TestCase):
    def dry_run(self, script_name):
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-File",
                str(ROOT / "ps_ablation" / script_name),
                "-Seeds",
                "42",
                "-DryRun",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def test_baseline_invokes_root_main_torch_with_baseline_architecture(self):
        dry_run = self.dry_run("01_baseline.ps1")
        invocation = dry_run["train"]
        expected_python = str(ROOT.parent / ".venv" / "Scripts" / "python.exe")
        if not Path(expected_python).is_file():
            expected_python = str(ROOT / ".venv" / "Scripts" / "python.exe")
        if not Path(expected_python).is_file():
            expected_python = "python"
        self.assertEqual(invocation[0], expected_python)
        self.assertEqual(invocation[1], str(ROOT / "main_torch.py"))
        self.assertIn("--enable_msc", invocation)
        self.assertEqual(invocation[invocation.index("--enable_msc") + 1], "False")
        self.assertEqual(invocation[invocation.index("--skip_mode") + 1], "normal")
        self.assertEqual(invocation[invocation.index("--seed") + 1], "42")

    def test_full_model_selects_all_architecture_components(self):
        invocation = self.dry_run("06_full_model.ps1")["train"]
        expected = {
            "--enable_msc": "True",
            "--skip_mode": "bsei",
            "--detail_channels": "32",
            "--enable_gdf": "True",
            "--detail_fusion_mode": "gdf",
            "--deep_supervision_heads": "3",
        }
        for option, value in expected.items():
            self.assertEqual(invocation[invocation.index(option) + 1], value)

    def test_launcher_enables_automatic_best_checkpoint_resume(self):
        invocation = self.dry_run("01_baseline.ps1")["train"]
        self.assertEqual(invocation[invocation.index("--auto_resume") + 1], "True")
        self.assertEqual(invocation[invocation.index("--resume_optimizer") + 1], "True")

    def test_dry_run_contains_train_evaluate_and_summarize_commands(self):
        dry_run = self.dry_run("01_baseline.ps1")
        self.assertEqual(Path(dry_run["evaluate"][1]).name, "evaluate.py")
        self.assertEqual(Path(dry_run["summarize"][1]).name, "summarize_seeds.py")

    def test_main_torch_ablation_arguments_parse_launcher_values(self):
        parser = argparse.ArgumentParser()
        add_ablation_arguments(parser)
        args = parser.parse_args([
            "--seed", "42",
            "--encoder_weights", "weights.pth",
            "--enable_msc", "False",
            "--skip_mode", "bsei",
            "--detail_channels", "32",
            "--enable_gdf", "True",
            "--detail_fusion_mode", "gdf",
            "--deep_supervision_heads", "3",
        ])
        self.assertEqual(args.seed, 42)
        self.assertFalse(args.enable_msc)
        self.assertEqual(args.skip_mode, "bsei")
        self.assertEqual(args.detail_channels, 32)
        self.assertTrue(args.enable_gdf)
        self.assertEqual(args.detail_fusion_mode, "gdf")
        self.assertEqual(args.deep_supervision_heads, 3)


if __name__ == "__main__":
    unittest.main()
