import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import build_optimizer_param_groups
from one_seed_models import build_experiment_model, main_logits


REFERENCE = "one_seed_37_fafem_mscb_lite_stage3_warmup_cosine"
EXPERIMENT = "one_seed_41_fafem_mscb_lite_stage3_stage2_warmup_cosine"
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "41_fafem_mscb_lite_stage3_stage2_warmup_cosine_seed42.ps1"


class FAFEMMSCBLiteStage2Tests(unittest.TestCase):
    def test_registered_experiment_differs_from_stage3_reference_only_by_stage2_mscb(self):
        reference = get_experiment(REFERENCE)
        candidate = get_experiment(EXPERIMENT)

        reference_values = reference.to_dict()
        candidate_values = candidate.to_dict()
        reference_values.pop("name")
        candidate_values.pop("name")
        reference_values["enable_mscb_lite_stage2"] = False
        self.assertEqual(candidate_values, {
            **reference_values,
            "enable_mscb_lite_stage2": True,
        })

    def test_stage2_mscb_preserves_shape_and_forward_output_shape(self):
        config = get_experiment(EXPERIMENT)
        model = build_experiment_model(config, encoder_weights=None).eval()

        self.assertIsNotNone(model.mscb_lite_stage3)
        self.assertIsNotNone(model.mscb_lite_stage2)
        with torch.no_grad():
            self.assertEqual(
                tuple(model.mscb_lite_stage2(torch.randn(1, 192, 44, 44)).shape),
                (1, 192, 44, 44),
            )
            logits = main_logits(model(torch.randn(1, 3, 352, 352)))
        self.assertEqual(tuple(logits.shape), (1, 1, 352, 352))

    def test_stage2_mscb_parameters_are_trainable_decoder_parameters(self):
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
        stage2_params = {id(parameter) for parameter in model.mscb_lite_stage2.parameters()}
        decoder_group = next(group for group in groups if group["name"] == "decoder")
        decoder_params = {id(parameter) for parameter in decoder_group["params"]}

        self.assertEqual(sum(parameter.numel() for parameter in model.mscb_lite_stage2.parameters()), 164_352)
        self.assertTrue(all(parameter.requires_grad for parameter in model.mscb_lite_stage2.parameters()))
        self.assertTrue(stage2_params.issubset(decoder_params))

    def test_runner_dry_run_matches_experiment_37_protocol_plus_stage2_mscb(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = next(json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{"))
        command = payload["train"]
        arguments = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(arguments["--experiment_name"], EXPERIMENT)
        self.assertEqual(arguments["--seed"], "42")
        self.assertEqual(arguments["--enable_fafem"], "True")
        self.assertEqual(arguments["--enable_mscb_lite_stage3"], "True")
        self.assertEqual(arguments["--enable_mscb_lite_stage2"], "True")
        self.assertEqual(arguments["--enable_lka_lite_stage3"], "False")
        self.assertEqual(arguments["--tta_check"], "False")


if __name__ == "__main__":
    unittest.main()
