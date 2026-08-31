import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from models.convnext_pretrain import MixStyle
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
EXP45 = "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"
EXP52 = "one_seed_52_fafem_residual_frequency_guided_mscb_stage3_mixstyle12_warmup_cosine"
RUNNER = ROOT / "ps_one_seed_ablation" / "52_fafem_residual_frequency_guided_mscb_stage3_mixstyle12_warmup_cosine_seed42.ps1"


class MixStyleStage12Tests(unittest.TestCase):
    def test_mixstyle_is_active_only_in_training_mode(self):
        features = torch.stack([
            torch.full((96, 8, 8), float(value)) + torch.arange(8).view(1, 8, 1)
            for value in range(4)
        ])
        module = MixStyle(probability=1.0, alpha=0.1)

        torch.manual_seed(42)
        module.train()
        mixed = module(features)
        self.assertEqual(mixed.shape, features.shape)
        self.assertFalse(torch.equal(mixed, features))

        module.eval()
        self.assertTrue(torch.equal(module(features), features))

    def test_exp52_only_adds_stage12_mixstyle_to_exp45(self):
        reference = get_experiment(EXP45).to_dict()
        candidate = get_experiment(EXP52).to_dict()
        self.assertTrue(candidate.pop("enable_mixstyle_stage1_stage2"))
        for item in (reference, candidate):
            item.pop("name")
        self.assertEqual(reference, candidate)

    def test_model_routes_mixstyle_only_through_stage1_and_stage2(self):
        model = build_experiment_model(get_experiment(EXP52), encoder_weights=None).eval()
        self.assertIsNotNone(model.mixstyle_stage1)
        self.assertIsNotNone(model.mixstyle_stage2)
        self.assertIsNone(getattr(model, "mixstyle_stage3", None))
        self.assertIsNone(getattr(model, "mixstyle_stage4", None))
        with torch.no_grad():
            output = model(torch.randn(1, 3, 352, 352))
        self.assertEqual(output.shape, (1, 1, 352, 352))

    def test_runner_dry_run_preserves_exp45_protocol(self):
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        command = dict(zip(json.loads(result.stdout.strip())["train"][::2], json.loads(result.stdout.strip())["train"][1::2]))
        self.assertEqual(command["--experiment_name"], EXP52)
        self.assertEqual(command["--seed"], "42")
        self.assertEqual(command["--batch_size"], "24")
        self.assertEqual(command["--enable_mixstyle_stage1_stage2"], "True")
        self.assertEqual(command["--enable_residual_fg_mscb_lite_stage3"], "True")
        self.assertEqual(command["--tta_check"], "False")


if __name__ == "__main__":
    unittest.main()
