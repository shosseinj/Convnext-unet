import json
import os
import shutil
import subprocess
import tempfile
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

    def test_default_three_seed_dry_run_never_executes_aggregation(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_ablation" / "01_baseline.ps1"), "-DryRun",
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        self.assertEqual([item["train"][item["train"].index("--seed") + 1] for item in payloads],
                         ["42", "6543", "7777"])

    def test_runner_finishes_each_seed_evaluation_before_advancing_and_summarizes_last(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            repo = project / "repo"
            scripts = repo / "ps_ablation"
            scripts.mkdir(parents=True)
            shutil.copy(ROOT / "ps_ablation" / "Invoke-Ablation.ps1", scripts)
            shutil.copy(ROOT / "ps_ablation" / "01_baseline.ps1", scripts)
            (repo / "convnext_tiny_22k_1k_384.pth").write_bytes(b"fixture")
            log_path = project / "operations.log"

            common = """import argparse, os\nfrom pathlib import Path\ndef log(value):\n    with open(os.environ['ABLATION_TEST_LOG'], 'a', encoding='utf-8') as handle:\n        handle.write(value + '\\n')\n"""
            (repo / "ablation_state.py").write_text(common + """
import json
p=argparse.ArgumentParser(); p.add_argument('--seed', type=int); p.add_argument('--seed_dir', type=Path); p.add_argument('--experiment_name'); p.add_argument('--max_epochs'); p.add_argument('--log_path'); a=p.parse_args()
log(f'state:{a.seed}')
trained=(a.seed_dir/'trained').exists(); evaluated=(a.seed_dir/'evaluated').exists()
print(json.dumps({'training_action':'skip' if trained else 'train','training_reason':'fixture','evaluation_valid':trained and evaluated,'evaluation_reason':'fixture'}))
""", encoding="utf-8")
            (repo / "main_torch.py").write_text(common + """
p=argparse.ArgumentParser(); p.add_argument('--seed', type=int); p.add_argument('--seed_dir', type=Path); a,_=p.parse_known_args(); log(f'train:{a.seed}'); a.seed_dir.mkdir(parents=True, exist_ok=True); (a.seed_dir/'trained').touch()
""", encoding="utf-8")
            (repo / "evaluate.py").write_text(common + """
p=argparse.ArgumentParser(); p.add_argument('--seed', type=int); p.add_argument('--seed_dir', type=Path); a,_=p.parse_known_args(); log(f'evaluate:{a.seed}'); (a.seed_dir/'evaluated').touch()
""", encoding="utf-8")
            (repo / "summarize_seeds.py").write_text(common + "log('summarize')\n", encoding="utf-8")
            seed_42 = repo / "results" / "ablation" / "01_baseline" / "seed_42"
            seed_42.mkdir(parents=True)
            (seed_42 / "trained").touch()

            environment = dict(os.environ, ABLATION_TEST_LOG=str(log_path))
            subprocess.run(
                ["powershell", "-NoProfile", "-File", str(scripts / "01_baseline.ps1")],
                cwd=repo, env=environment, check=True, capture_output=True, text=True,
            )
            operations = log_path.read_text(encoding="utf-8").splitlines()
            significant = [entry for entry in operations if not entry.startswith("state:")]
            self.assertEqual(significant, [
                "evaluate:42", "train:6543", "evaluate:6543",
                "train:7777", "evaluate:7777", "summarize",
            ])

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

    def test_main_torch_ablation_artifact_arguments_are_registered(self):
        parser = argparse.ArgumentParser()
        add_ablation_arguments(parser)
        args = parser.parse_args([
            "--experiment_name", "01_baseline",
            "--seed_dir", "seed_42",
            "--best_checkpoint_path", "seed_42/best_checkpoint.pth",
            "--training_history_path", "seed_42/training_history.csv",
            "--training_summary_path", "seed_42/training_summary.json",
        ])
        self.assertEqual(args.experiment_name, "01_baseline")
        self.assertEqual(args.seed_dir, "seed_42")
        self.assertEqual(args.best_checkpoint_path, "seed_42/best_checkpoint.pth")


if __name__ == "__main__":
    unittest.main()
