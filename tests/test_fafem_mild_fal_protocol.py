import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import frequency_augmentation_probability, frequency_style_augment
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "one_seed_31_fafem_mild_fal"


class FafemMildFalProtocolTests(unittest.TestCase):
    def test_registry_differs_from_fafem_only_only_by_fal_metadata(self):
        reference = get_experiment("one_seed_03_baseline_plus_fafem")
        candidate = get_experiment(EXPERIMENT)
        ignored = {"name", "enable_frequency_augmentation"}
        for field in reference.__dataclass_fields__:
            if field not in ignored:
                self.assertEqual(getattr(candidate, field), getattr(reference, field), field)
        self.assertTrue(candidate.enable_frequency_augmentation)
        self.assertEqual(candidate.uncertainty_refinement_version, "none")

    def test_model_is_the_same_fafem_only_architecture(self):
        reference = build_experiment_model(
            get_experiment("one_seed_03_baseline_plus_fafem"), None
        )
        candidate = build_experiment_model(get_experiment(EXPERIMENT), None)
        self.assertEqual(reference.state_dict().keys(), candidate.state_dict().keys())
        self.assertFalse(hasattr(candidate, "uncertainty_refinement"))

    def test_disabled_fal_is_identity_and_mild_fal_is_reproducible(self):
        images = torch.rand(4, 3, 352, 352)
        identity = frequency_style_augment(
            images, 0.0, torch.Generator().manual_seed(42),
            max_mix=0.25, region_min=0.01, region_max=0.03,
        )
        self.assertTrue(torch.equal(identity, images))
        generators = [torch.Generator().manual_seed(42) for _ in range(2)]
        outputs = [
            frequency_style_augment(
                images, 1.0, generator, max_mix=0.25,
                region_min=0.01, region_max=0.03,
            ) for generator in generators
        ]
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertEqual(outputs[0].shape, images.shape)
        self.assertGreaterEqual(outputs[0].min().item(), 0.0)
        self.assertLessEqual(outputs[0].max().item(), 1.0)

    def test_mild_probability_schedule(self):
        kwargs = dict(
            total_epochs=350, maximum=0.25,
            constant_fraction=0.60, anneal_end_fraction=0.70,
        )
        self.assertEqual(frequency_augmentation_probability(1, **kwargs), 0.25)
        self.assertEqual(frequency_augmentation_probability(210, **kwargs), 0.25)
        self.assertAlmostEqual(frequency_augmentation_probability(227, **kwargs), 0.1285714286)
        self.assertEqual(frequency_augmentation_probability(245, **kwargs), 0.0)
        self.assertEqual(frequency_augmentation_probability(350, **kwargs), 0.0)

    def test_runner_dry_run_is_isolated_and_matches_reference_protocol(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(
                ROOT / "ps_one_seed_ablation" / "31_fafem_mild_fal_seed42.ps1"
            ), "-DryRun"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        payload = json.loads(next(
            line for line in completed.stdout.splitlines() if line.startswith("{")
        ))
        train = payload["train"]
        argument = lambda flag: train[train.index(flag) + 1]
        expected = {
            "--experiment_name": EXPERIMENT,
            "--seed": "42", "--batch_size": "24", "--epochs": "350",
            "--decoder_warmup_epochs": "15",
            "--unfreeze_plateau_patience": "8",
            "--lr_plateau_patience": "12", "--lr_scheduler": "plateau",
            "--enable_fafem": "True", "--enable_frequency_augmentation": "True",
            "--frequency_max_probability": "0.25", "--frequency_max_mix": "0.25",
            "--frequency_region_min": "0.01", "--frequency_region_max": "0.03",
            "--frequency_constant_fraction": "0.6",
            "--frequency_anneal_end_fraction": "0.7",
            "--uncertainty_refinement_version": "none",
        }
        for flag, value in expected.items():
            self.assertEqual(argument(flag), value, flag)
        self.assertIn("31_fafem_mild_fal", payload["seed_dir"])


if __name__ == "__main__":
    unittest.main()
