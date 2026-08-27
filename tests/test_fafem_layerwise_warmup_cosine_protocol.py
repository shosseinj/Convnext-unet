import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import build_optimizer_param_groups, create_warmup_cosine_scheduler
from one_seed_models import build_experiment_model, main_logits


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "one_seed_32_fafem_layerwise_warmup_cosine"
REFERENCE = "one_seed_03_baseline_plus_fafem"


class FAFEMLayerwiseWarmupCosineProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_experiment_model(get_experiment(EXPERIMENT), None)

    def test_registry_is_fafem_only_with_training_recipe_change(self):
        candidate = get_experiment(EXPERIMENT).to_dict()
        reference = get_experiment(REFERENCE).to_dict()
        self.assertEqual(candidate.pop("unfreeze_schedule"), "none")
        self.assertNotIn("unfreeze_schedule", reference)
        for key in ("name", "max_epochs", "encoder_freeze_epochs"):
            candidate.pop(key)
            reference.pop(key)
        self.assertEqual(candidate, reference)

    def test_model_preserves_shape(self):
        self.model.eval()
        with torch.no_grad():
            output = main_logits(self.model(torch.randn(1, 3, 352, 352)))
        self.assertEqual(tuple(output.shape), (1, 1, 352, 352))

    def test_layerwise_optimizer_groups_are_exhaustive(self):
        groups = build_optimizer_param_groups(
            self.model,
            decoder_lr=3e-4,
            encoder_lr=4e-5,
            refine_lr=4.5e-4,
            decoder_weight_decay=1e-4,
            encoder_weight_decay=5e-2,
            refine_weight_decay=1e-2,
            profile="layerwise_convnext",
            encoder_layer_decay=0.8,
        )
        expected = {"refine", "decoder", "encoder_stage4", "encoder_stage3", "encoder_stage2", "encoder_stage1_stem"}
        self.assertEqual({group["name"] for group in groups}, expected)
        self.assertAlmostEqual(next(group["lr"] for group in groups if group["name"] == "encoder_stage3"), 4e-5 * 0.8)

    def test_warmup_cosine_scheduler_hits_min_lr(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([{"name": "decoder", "params": [parameter], "lr": 3e-4}])
        scheduler = create_warmup_cosine_scheduler(optimizer, 5, 160, 1e-6)
        for _ in range(160):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)

    def test_runner_dry_run(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" / "32_fafem_layerwise_warmup_cosine_seed42.ps1"),
                "-DryRun",
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        payload = json.loads(next(line for line in completed.stdout.splitlines() if line.startswith("{")))
        train = payload["train"]
        self.assertEqual(payload["experiment"], EXPERIMENT)
        expected = {
            "--seed": "42",
            "--batch_size": "24",
            "--epochs": "160",
            "--unfreeze_schedule": "none",
            "--lr_scheduler": "warmup_cosine",
            "--warmup_epochs": "5",
            "--optimizer_profile": "layerwise_convnext",
            "--encoder_layer_decay": "0.8",
            "--enable_fafem": "True",
            "--max_grad_norm": "1",
        }
        for flag, value in expected.items():
            self.assertEqual(train[train.index(flag) + 1], value, flag)


if __name__ == "__main__":
    unittest.main()
