import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import (
    build_optimizer_param_groups,
    frequency_augmentation_probability,
    frequency_style_augment,
    set_training_stage,
)
from one_seed_models import build_experiment_model
from research_pipeline.losses import ugbr_composite_loss


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "one_seed_30_fafem_fal_uncertainty_refinement"


class FalUncertaintyRefinementProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_experiment_model(get_experiment(EXPERIMENT), None)

    def test_registered_architecture_is_isolated(self):
        config = get_experiment(EXPERIMENT)
        self.assertTrue(config.enable_fafem)
        self.assertTrue(config.enable_frequency_augmentation)
        self.assertEqual(config.uncertainty_refinement_version, "v2")
        self.assertFalse(config.enable_msc)
        self.assertFalse(config.enable_ugbr)
        self.assertFalse(config.enable_cross_level_fusion)
        self.assertEqual(config.detail_channels, 0)
        self.assertEqual(config.deep_supervision_heads, 0)

    def test_refinement_is_identity_safe_and_shape_preserving(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.randn(1, 3, 352, 352))
        self.assertEqual(tuple(output["final_logits"].shape), (1, 1, 352, 352))
        self.assertTrue(torch.equal(output["final_logits"], output["initial_logits"]))
        self.assertTrue(torch.count_nonzero(output["refinement_logits"]) == 0)

    def test_composite_loss_trains_output_and_boundary_heads(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), None).train()
        output = model(torch.randn(1, 3, 64, 64))
        target = torch.randint(0, 2, output["final_logits"].shape).float()
        components = ugbr_composite_loss(
            output, target, torch.nn.BCEWithLogitsLoss()
        )
        components["total"].backward()
        refinement = model.uncertainty_refinement
        self.assertIsNotNone(refinement.refinement_output.weight.grad)
        self.assertGreater(refinement.refinement_output.weight.grad.abs().sum(), 0)
        self.assertTrue(any(p.grad is not None for p in refinement.boundary_head.parameters()))

    def test_frequency_augmentation_is_bounded_and_reproducible(self):
        images = torch.rand(4, 3, 32, 32)
        generators = [torch.Generator().manual_seed(72) for _ in range(2)]
        outputs = [frequency_style_augment(images, 1.0, generator) for generator in generators]
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertEqual(outputs[0].shape, images.shape)
        self.assertGreaterEqual(outputs[0].min().item(), 0.0)
        self.assertLessEqual(outputs[0].max().item(), 1.0)
        self.assertFalse(torch.equal(outputs[0], images))

    def test_frequency_probability_schedule(self):
        self.assertEqual(frequency_augmentation_probability(1, 180), 0.5)
        self.assertEqual(frequency_augmentation_probability(126, 180), 0.5)
        self.assertAlmostEqual(frequency_augmentation_probability(135, 180), 0.25)
        self.assertEqual(frequency_augmentation_probability(144, 180), 0.0)
        self.assertEqual(frequency_augmentation_probability(180, 180), 0.0)

    def test_new_module_is_decoder_group_and_trainable_during_warmup(self):
        set_training_stage(self.model, "decoder")
        self.assertFalse(any(p.requires_grad for p in self.model.encoder.parameters()))
        self.assertTrue(all(
            p.requires_grad for p in self.model.uncertainty_refinement.parameters()
        ))
        groups = build_optimizer_param_groups(
            self.model, 3e-4, 3e-5, 4.5e-4, 1e-2, 5e-2, 1e-2,
            profile="layerwise_convnext", encoder_layer_decay=0.8,
        )
        decoder_ids = {
            id(parameter) for group in groups if group["name"] == "decoder"
            for parameter in group["params"]
        }
        self.assertTrue(all(
            id(parameter) in decoder_ids
            for parameter in self.model.uncertainty_refinement.parameters()
        ))
        set_training_stage(self.model, "all")
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_runner_dry_run(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-File",
             str(ROOT / "ps_one_seed_ablation" /
                 "30_fafem_fal_uncertainty_refinement_seed42.ps1"), "-DryRun"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        payload = json.loads(next(
            line for line in completed.stdout.splitlines() if line.startswith("{")
        ))
        train = payload["train"]
        argument = lambda flag: train[train.index(flag) + 1]
        expected = {
            "--seed": "42", "--batch_size": "32", "--epochs": "180",
            "--decoder_warmup_epochs": "5", "--lr_scheduler": "warmup_cosine",
            "--warmup_epochs": "5", "--enable_fafem": "True",
            "--enable_frequency_augmentation": "True",
            "--uncertainty_refinement_version": "v2",
            "--early_stop_start_epoch": "144", "--early_stop_patience": "35",
        }
        for flag, value in expected.items():
            self.assertEqual(argument(flag), value, flag)


if __name__ == "__main__":
    unittest.main()
