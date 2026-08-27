import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import build_optimizer_param_groups, set_training_stage
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "one_seed_fafem_plus_geometry_conv_stage3"


class FAFEMGeometryConvStage3AblationTests(unittest.TestCase):
    def test_registry_enables_only_geometry_delta(self):
        reference = get_experiment("one_seed_03_baseline_plus_fafem").to_dict()
        candidate = get_experiment(EXPERIMENT).to_dict()
        self.assertTrue(candidate.pop("enable_geometry_conv_stage3"))
        self.assertNotIn("enable_geometry_conv_stage3", reference)
        for field in ("name", "max_epochs", "encoder_freeze_epochs", "unfreeze_schedule"):
            candidate.pop(field, None)
            reference.pop(field, None)
        self.assertEqual(candidate, reference)

    def test_model_contains_geometry_module_and_preserves_shape(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), None).eval()
        self.assertTrue(hasattr(model, "geometry_conv_stage3"))
        self.assertIsNotNone(model.geometry_conv_stage3)
        with torch.no_grad():
            output = model(torch.randn(1, 3, 352, 352))
        self.assertEqual(tuple(output.shape), (1, 1, 352, 352))

    def test_geometry_stage_is_trainable_from_epoch_one(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), None)
        set_training_stage(model, "all")
        groups = build_optimizer_param_groups(
            model,
            decoder_lr=3e-4,
            encoder_lr=4e-5,
            refine_lr=3e-4,
            decoder_weight_decay=1e-4,
            encoder_weight_decay=1e-4,
            refine_weight_decay=1e-4,
            profile="layerwise_convnext",
            encoder_layer_decay=0.75,
        )
        decoder_params = {
            id(param)
            for group in groups if group["name"] == "decoder"
            for param in group["params"]
        }
        self.assertTrue(
            all(id(param) in decoder_params for param in model.geometry_conv_stage3.parameters())
        )

    def test_runner_dry_run(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" / "run_fafem_plus_geometry_seed6543.ps1"),
                "-DryRun",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(next(line for line in completed.stdout.splitlines() if line.startswith("{")))
        train = payload["train"]
        self.assertEqual(payload["experiment"], EXPERIMENT)
        self.assertEqual(payload["output_name"], "fafem_plus_geometry_conv")
        self.assertIn("fafem_plus_geometry_conv", payload["seed_dir"])
        expected = {
            "--seed": "6543",
            "--batch_size": "24",
            "--epochs": "120",
            "--lr_scheduler": "warmup_cosine",
            "--warmup_epochs": "5",
            "--optimizer_profile": "layerwise_convnext",
            "--encoder_layer_decay": "0.75",
            "--max_grad_norm": "1",
            "--enable_fafem": "True",
            "--enable_geometry_conv_stage3": "True",
            "--unfreeze_schedule": "none",
            "--decoder_warmup_epochs": "0",
        }
        for flag, value in expected.items():
            self.assertEqual(train[train.index(flag) + 1], value, flag)


if __name__ == "__main__":
    unittest.main()
