import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from one_seed_models import (
    DySample, build_experiment_model, cosine_schedule_epochs, gradient_rms, main_logits,
)
from compare_one_seed import compare_experiments


ROOT = Path(__file__).resolve().parents[1]


class OneSeedAblationTests(unittest.TestCase):
    def test_registry_defines_isolated_ranked_experiments(self):
        expected = {
            "one_seed_01_baseline": ("normal", 0, False, "bilinear"),
            "one_seed_02_add_ugbr": ("normal", 0, True, "bilinear"),
            "one_seed_03_gated_skips": ("attention_gate", 0, False, "bilinear"),
            "one_seed_04_deep_supervision": ("normal", 2, False, "bilinear"),
            "one_seed_05_dysample": ("normal", 0, False, "dysample"),
        }
        for name, values in expected.items():
            config = get_experiment(name)
            self.assertEqual(
                (config.skip_mode, config.deep_supervision_heads,
                 config.enable_ugbr, config.upsample_mode), values
            )
            self.assertEqual(config.backbone, "convnext_tiny")

    def test_dysample_doubles_spatial_resolution(self):
        layer = DySample(32, scale=2, groups=4)
        output = layer(torch.randn(2, 32, 7, 11))
        self.assertEqual(output.shape, (2, 32, 14, 22))

    def test_model_builder_keeps_convnext_encoder_for_every_experiment(self):
        for name in (
            "one_seed_01_baseline", "one_seed_02_add_ugbr",
            "one_seed_03_gated_skips", "one_seed_04_deep_supervision",
            "one_seed_05_dysample",
        ):
            model = build_experiment_model(get_experiment(name), encoder_weights=None)
            self.assertEqual(model.variant_config["backbone"], "convnext_tiny")
            self.assertTrue(hasattr(model, "encoder"))

    def test_main_logits_selects_final_ugbr_prediction(self):
        final = torch.randn(1, 1, 8, 8)
        self.assertIs(main_logits({"initial_logits": torch.zeros_like(final),
                                  "final_logits": final}), final)

    def test_gradient_rms_normalizes_for_parameter_count(self):
        small = torch.nn.Parameter(torch.zeros(4)); small.grad = torch.ones(4)
        large = torch.nn.Parameter(torch.zeros(400)); large.grad = torch.ones(400)
        self.assertAlmostEqual(gradient_rms([small]), 1.0)
        self.assertAlmostEqual(gradient_rms([large]), 1.0)

    def test_cosine_schedule_covers_only_epochs_after_warmup(self):
        self.assertEqual(cosine_schedule_epochs(150, 0, 10), 140)
        self.assertEqual(cosine_schedule_epochs(5, 3, 3), 1)

    def test_all_runner_dry_run_uses_seed_42_and_isolated_results(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-File",
             str(ROOT / "ps_one_seed_ablation" / "Run-All.ps1"), "-DryRun"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(len(payloads), 5)
        for payload in payloads:
            train = payload["train"]
            self.assertEqual(train[train.index("--seed") + 1], "42")
            self.assertIn("one_seed_results", payload["seed_dir"])
            self.assertEqual(train[train.index("--batch_size") + 1], "24")
            self.assertEqual(train[train.index("--focal_tversky_after_warmup") + 1], "False")

    def test_comparison_is_written_only_when_every_evaluation_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(("01_baseline", "02_add_ugbr")):
                seed_dir = root / name / "seed_42"
                seed_dir.mkdir(parents=True)
                results = {
                    dataset: {metric: 0.8 + index * 0.01 for metric in
                              ("mDice", "mIoU", "F_beta_w", "S_alpha", "mE_phi", "maxE_phi", "MAE")}
                    for dataset in ("Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")
                }
                (seed_dir / "evaluation_summary.json").write_text(json.dumps({
                    "experiment_name": f"one_seed_{name}", "seed": 42,
                    "trainable_parameters": 10, "total_parameters": 12,
                    "macs": 1, "flops": 2, "gmacs": 1e-9, "gflops": 2e-9,
                    "results": results,
                }), encoding="utf-8")
            output = compare_experiments(root)
            self.assertTrue(output.is_file())
            rows = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            self.assertIn("mean_mDice", rows[0])


if __name__ == "__main__":
    unittest.main()
