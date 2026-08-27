import json
import subprocess
import unittest
from pathlib import Path

import numpy as np
import torch

from ablation_registry import get_experiment
from main_torch import (
    deep_supervision_weights_for_epoch,
    lesion_size_sampling_weights,
)
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
EXP28 = "one_seed_28_fafem_detail_clfv2_ds_anneal_layerwise_cosine"
EXP29 = "one_seed_29_fafem_detail_clfv2_ds_anneal_weighted"


class DeepSupervisionAnnealingProtocolTests(unittest.TestCase):
    def test_schedule_boundaries(self):
        self.assertEqual(deep_supervision_weights_for_epoch(96, "anneal"), (1.0, 0.1, 0.05, 0.02))
        self.assertEqual(deep_supervision_weights_for_epoch(112, "anneal"), (1.0, 0.05, 0.025, 0.01))
        self.assertEqual(deep_supervision_weights_for_epoch(128, "anneal"), (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(deep_supervision_weights_for_epoch(129, "anneal"), (1.0, 0.0, 0.0, 0.0))

    def test_sampling_buckets_and_generator_reproducibility(self):
        masks = np.zeros((3, 1, 10, 10), dtype=np.float32)
        masks[0, 0, 0, 0] = 1
        masks[1, 0, :2, :2] = 1
        masks[2, 0, :5, :5] = 1
        weights, _ = lesion_size_sampling_weights(masks)
        self.assertEqual(weights.tolist(), [4.0, 2.0, 1.0])
        generators = [torch.Generator().manual_seed(42) for _ in range(2)]
        draws = [
            list(torch.utils.data.WeightedRandomSampler(weights, 20, True, generator=g))
            for g in generators
        ]
        self.assertEqual(draws[0], draws[1])

    def test_forward_shapes(self):
        model = build_experiment_model(get_experiment(EXP28), None).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 352, 352))
        self.assertEqual([tuple(x.shape) for x in outputs], [(1, 1, 352, 352)] * 3)

    def test_runners_are_isolated(self):
        runners = {
            "28_fafem_detail_clfv2_ds_anneal_layerwise_cosine_seed42.ps1": (EXP28, "uniform"),
            "29_fafem_detail_clfv2_ds_anneal_weighted_seed42.ps1": (EXP29, "lesion_size_weighted"),
        }
        for runner, (experiment, sampling_mode) in runners.items():
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-File", str(ROOT / "ps_one_seed_ablation" / runner), "-DryRun"],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            payload = json.loads(next(line for line in completed.stdout.splitlines() if line.startswith("{")))
            train = payload["train"]
            argument = lambda flag: train[train.index(flag) + 1]
            self.assertEqual(payload["experiment"], experiment)
            self.assertEqual(argument("--sampling_mode"), sampling_mode)
            self.assertEqual(argument("--deep_supervision_schedule"), "anneal")
            self.assertEqual(argument("--epochs"), "160")
            self.assertEqual(argument("--seed"), "42")


if __name__ == "__main__":
    unittest.main()
