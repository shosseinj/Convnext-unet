import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import build_optimizer_param_groups
from one_seed_models import build_experiment_model, main_logits


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "one_seed_37_fafem_mscb_lite_stage3_warmup_cosine"
EXPERIMENT = "one_seed_39_fafem_mscb_lite_detail_warmup_cosine"
RUNNER = ROOT / "ps_one_seed_ablation" / "39_fafem_mscb_lite_detail_warmup_cosine.ps1"


class FAFEMMSCBDetailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = get_experiment(EXPERIMENT)
        cls.model = build_experiment_model(cls.config, None)

    def test_registry_differs_from_experiment_37_only_by_detail_branch(self):
        reference = get_experiment(REFERENCE).to_dict()
        candidate = self.config.to_dict()
        self.assertEqual(reference.pop("name"), REFERENCE)
        self.assertEqual(candidate.pop("name"), EXPERIMENT)
        self.assertEqual(reference.pop("detail_channels"), 0)
        self.assertEqual(reference.pop("detail_fusion_mode"), "none")
        self.assertEqual(candidate.pop("detail_channels"), 32)
        self.assertEqual(candidate.pop("detail_fusion_mode"), "concatenation")
        self.assertEqual(candidate, reference)

    def test_forward_shape_and_detail_parameters_are_decoder_trainable(self):
        self.model.eval()
        with torch.no_grad():
            logits = main_logits(self.model(torch.randn(1, 3, 352, 352)))
        self.assertEqual(tuple(logits.shape), (1, 1, 352, 352))

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
        decoder_ids = {id(parameter) for group in groups if group["name"] == "decoder" for parameter in group["params"]}
        detail_parameters = [parameter for name, parameter in self.model.named_parameters() if name.startswith(("detail_branch.", "detail_fusion."))]
        self.assertTrue(detail_parameters)
        self.assertTrue(all(parameter.requires_grad for parameter in detail_parameters))
        self.assertTrue(all(id(parameter) in decoder_ids for parameter in detail_parameters))

    def test_three_seed_runner_dry_run_preserves_experiment_37_protocol(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
        self.assertEqual([payload["experiment"] for payload in payloads], [EXPERIMENT] * 3)
        self.assertEqual([payload["train"][payload["train"].index("--seed") + 1] for payload in payloads], ["42", "6543", "7777"])
        for payload in payloads:
            train = payload["train"]
            expected = {
                "--detail_channels": "32",
                "--detail_fusion_mode": "concatenation",
                "--enable_fafem": "True",
                "--enable_mscb_lite_stage3": "True",
                "--enable_msc": "False",
                "--enable_ugbr": "False",
                "--enable_cross_level_fusion": "False",
                "--enable_gated_skip_stage3": "False",
                "--deep_supervision_heads": "0",
                "--lr_scheduler": "warmup_cosine",
                "--warmup_epochs": "5",
                "--optimizer_profile": "layerwise_convnext",
                "--tta_check": "False",
            }
            for flag, value in expected.items():
                self.assertEqual(train[train.index(flag) + 1], value, flag)


if __name__ == "__main__":
    unittest.main()
