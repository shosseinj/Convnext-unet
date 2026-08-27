import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import (
    build_optimizer_param_groups,
    create_warmup_cosine_scheduler,
)
from one_seed_models import build_experiment_model, main_logits


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "one_seed_26_fafem_detail_clfv2_layerwise_cosine"
REFERENCE = "one_seed_25_fafem_detail_clfv2_warmup_cosine"


class LayerwiseCosineProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_experiment_model(get_experiment(EXPERIMENT), None)

    def test_registry_changes_only_training_schedule_identity(self):
        candidate = get_experiment(EXPERIMENT).to_dict()
        reference = get_experiment(REFERENCE).to_dict()
        self.assertEqual(candidate.pop("unfreeze_schedule"), "none")
        self.assertEqual(reference.pop("unfreeze_schedule"), "fixed")
        for key in ("name", "max_epochs", "encoder_freeze_epochs"):
            candidate.pop(key)
            reference.pop(key)
        self.assertEqual(candidate, reference)

    def test_layerwise_groups_are_exhaustive_and_have_expected_hyperparameters(self):
        groups = build_optimizer_param_groups(
            self.model,
            decoder_lr=3e-4,
            encoder_lr=3e-5,
            refine_lr=4.5e-4,
            decoder_weight_decay=1e-2,
            encoder_weight_decay=5e-2,
            refine_weight_decay=1e-2,
            profile="layerwise_convnext",
            encoder_layer_decay=0.8,
        )
        by_name = {group["name"]: group for group in groups}
        expected_lrs = {
            "refine": 4.5e-4,
            "decoder": 3e-4,
            "encoder_stage4": 3e-5,
            "encoder_stage3": 2.4e-5,
            "encoder_stage2": 1.92e-5,
            "encoder_stage1_stem": 1.536e-5,
        }
        self.assertEqual(set(by_name), set(expected_lrs))
        for name, expected_lr in expected_lrs.items():
            self.assertAlmostEqual(by_name[name]["lr"], expected_lr)
            expected_decay = 1e-2 if name in {"refine", "decoder"} else 5e-2
            self.assertEqual(by_name[name]["weight_decay"], expected_decay)
        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
        model_ids = [id(parameter) for parameter in self.model.parameters()]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(model_ids))

    def test_warmup_reaches_max_and_cosine_reaches_min(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([{"name": "decoder", "params": [parameter], "lr": 3e-4}])
        scheduler = create_warmup_cosine_scheduler(optimizer, 5, 160, 1e-6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3e-5)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3e-4)
        saved_optimizer = optimizer.state_dict()
        saved_scheduler = scheduler.state_dict()
        resumed_parameter = torch.nn.Parameter(torch.ones(1))
        resumed_optimizer = torch.optim.AdamW([
            {"name": "decoder", "params": [resumed_parameter], "lr": 3e-4}
        ])
        resumed_scheduler = create_warmup_cosine_scheduler(
            resumed_optimizer, 5, 160, 1e-6
        )
        resumed_optimizer.load_state_dict(saved_optimizer)
        resumed_scheduler.load_state_dict(saved_scheduler)
        self.assertAlmostEqual(
            resumed_optimizer.param_groups[0]["lr"],
            optimizer.param_groups[0]["lr"],
        )
        for _ in range(155):
            resumed_optimizer.step()
            resumed_scheduler.step()
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], 1e-6)

    def test_forward_shape_is_unchanged(self):
        self.model.eval()
        with torch.no_grad():
            output = main_logits(self.model(torch.randn(1, 3, 352, 352)))
        self.assertEqual(tuple(output.shape), (1, 1, 352, 352))

    def test_runner_dry_run_is_isolated_and_exact(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" /
                    "26_fafem_detail_clfv2_layerwise_cosine_seed42.ps1"),
                "-DryRun",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(next(
            line for line in completed.stdout.splitlines()
            if line.strip().startswith("{")
        ))
        train = payload["train"]
        argument = lambda flag: train[train.index(flag) + 1]
        self.assertEqual(payload["experiment"], EXPERIMENT)
        self.assertTrue(payload["seed_dir"].endswith(
            "26_fafem_detail_clfv2_layerwise_cosine\\seed_42"
        ))
        expected = {
            "--seed": "42",
            "--batch_size": "24",
            "--epochs": "160",
            "--unfreeze_schedule": "none",
            "--lr_scheduler": "warmup_cosine",
            "--warmup_epochs": "5",
            "--lr": "0.0003",
            "--min_lr": "1E-06",
            "--optimizer_profile": "layerwise_convnext",
            "--encoder_layer_decay": "0.8",
            "--weight_decay": "0.01",
            "--encoder_weight_decay": "0.05",
            "--new_layer_weight_decay": "0.01",
            "--max_grad_norm": "1",
            "--enable_fafem": "True",
            "--detail_channels": "32",
            "--enable_cross_level_fusion": "True",
            "--cross_level_fusion_version": "v2",
        }
        for flag, value in expected.items():
            self.assertEqual(argument(flag), value, flag)


if __name__ == "__main__":
    unittest.main()
