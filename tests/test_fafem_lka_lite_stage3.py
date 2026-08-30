import unittest
import json
import subprocess
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import build_optimizer_param_groups
from one_seed_models import build_experiment_model, main_logits


REFERENCE = "one_seed_37_fafem_mscb_lite_stage3_warmup_cosine"
EXPERIMENT = "one_seed_40_fafem_lka_lite_stage3_mscb_warmup_cosine"
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "40_fafem_lka_lite_stage3_mscb_warmup_cosine_seed42.ps1"


class FAFEMLKALiteStage3Tests(unittest.TestCase):
    def test_registered_experiment_differs_from_mscb_reference_only_by_lka(self):
        reference = get_experiment(REFERENCE)
        candidate = get_experiment(EXPERIMENT)

        reference_values = reference.to_dict()
        candidate_values = candidate.to_dict()
        reference_values.pop("name")
        candidate_values.pop("name")
        reference_values["enable_lka_lite_stage3"] = False
        self.assertEqual(candidate_values, {
            **reference_values,
            "enable_lka_lite_stage3": True,
        })

    def test_zero_gamma_preserves_stage3_skip_and_main_output_shape(self):
        config = get_experiment(EXPERIMENT)
        torch.manual_seed(42)
        model = build_experiment_model(config, encoder_weights=None).eval()

        self.assertTrue(config.enable_lka_lite_stage3)
        self.assertIsNotNone(model.lka_lite_stage3)
        self.assertEqual(tuple(model.lka_lite_stage3.gamma.shape), (1,))
        self.assertEqual(float(model.lka_lite_stage3.gamma.item()), 0.0)

        skip = torch.randn(1, 384, 22, 22)
        with torch.no_grad():
            self.assertTrue(torch.equal(model.lka_lite_stage3(skip), skip))
            logits = main_logits(model(torch.randn(1, 3, 352, 352)))
        self.assertEqual(tuple(logits.shape), (1, 1, 352, 352))

    def test_lka_parameters_are_in_decoder_optimizer_group(self):
        config = get_experiment(EXPERIMENT)
        model = build_experiment_model(config, encoder_weights=None)
        groups = build_optimizer_param_groups(
            model,
            decoder_lr=3e-4,
            encoder_lr=4e-5,
            refine_lr=4.5e-4,
            decoder_weight_decay=1e-4,
            encoder_weight_decay=5e-2,
            refine_weight_decay=1e-2,
            profile="layerwise_convnext",
            encoder_layer_decay=0.8,
        )
        lka_params = {id(parameter) for parameter in model.lka_lite_stage3.parameters()}
        decoder_group = next(group for group in groups if group["name"] == "decoder")
        decoder_params = {id(parameter) for parameter in decoder_group["params"]}
        self.assertTrue(all(parameter.requires_grad for parameter in model.lka_lite_stage3.parameters()))
        self.assertTrue(lka_params.issubset(decoder_params))

    def test_runner_dry_run_covers_three_seeds_with_experiment_37_protocol_plus_lka(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payloads = [
            json.loads(line)
            for line in completed.stdout.splitlines()
            if line.startswith("{")
        ]
        self.assertEqual(len(payloads), 3)
        for expected_seed, payload in zip((42, 6543, 7777), payloads):
            command = payload["train"]
            arguments = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(arguments["--experiment_name"], EXPERIMENT)
            self.assertEqual(arguments["--seed"], str(expected_seed))
            self.assertEqual(arguments["--enable_fafem"], "True")
            self.assertEqual(arguments["--enable_mscb_lite_stage3"], "True")
            self.assertEqual(arguments["--enable_lka_lite_stage3"], "True")
            self.assertEqual(arguments["--tta_check"], "False")


if __name__ == "__main__":
    unittest.main()
