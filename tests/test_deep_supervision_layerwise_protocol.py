import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import deep_supervision_loss
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "one_seed_27_fafem_detail_clfv2_deep_supervision_layerwise_cosine"
REFERENCE = "one_seed_26_fafem_detail_clfv2_layerwise_cosine"


class DeepSupervisionLayerwiseProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_experiment_model(get_experiment(EXPERIMENT), None)

    def test_registry_diff_is_limited_to_deep_supervision_and_horizon(self):
        candidate = get_experiment(EXPERIMENT).to_dict()
        reference = get_experiment(REFERENCE).to_dict()
        self.assertEqual(candidate.pop("deep_supervision_heads"), 2)
        self.assertEqual(reference.pop("deep_supervision_heads"), 0)
        self.assertEqual(candidate.pop("max_epochs"), 220)
        self.assertEqual(reference.pop("max_epochs"), 160)
        candidate.pop("name")
        reference.pop("name")
        self.assertEqual(candidate, reference)

    def test_two_auxiliary_heads_preserve_segmentation_shape(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.randn(1, 3, 352, 352))
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(1, 1, 352, 352)] * 3,
        )

    def test_auxiliary_heads_receive_weighted_gradients(self):
        outputs = tuple(
            torch.zeros(1, 1, 16, 16, requires_grad=True)
            for _ in range(3)
        )
        target = torch.ones_like(outputs[0])
        _, loss = deep_supervision_loss(
            outputs, target, torch.nn.BCEWithLogitsLoss()
        )
        loss.backward()
        self.assertTrue(all(output.grad is not None for output in outputs))
        self.assertGreater(outputs[1].grad.abs().sum().item(), 0)
        self.assertGreater(outputs[2].grad.abs().sum().item(), 0)

    def test_runner_dry_run_has_exact_isolated_protocol(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" /
                    "27_fafem_detail_clfv2_deep_supervision_layerwise_cosine_seed42.ps1"),
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
            "27_fafem_detail_clfv2_deep_supervision_layerwise_cosine\\seed_42"
        ))
        expected = {
            "--seed": "42",
            "--batch_size": "24",
            "--epochs": "220",
            "--deep_supervision_heads": "2",
            "--unfreeze_schedule": "none",
            "--lr_scheduler": "warmup_cosine",
            "--warmup_epochs": "5",
            "--lr": "0.0003",
            "--min_lr": "1E-06",
            "--optimizer_profile": "layerwise_convnext",
            "--encoder_layer_decay": "0.8",
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
