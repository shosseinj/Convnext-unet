import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import build_optimizer_param_groups, summarize_fg_mscb_alpha
from models.convnext_pretrain import MSCBLite
from one_seed_models import build_experiment_model, main_logits


EXPERIMENT = "one_seed_43_fafem_frequency_guided_mscb_stage3_warmup_cosine"
REFERENCE = "one_seed_37_fafem_mscb_lite_stage3_warmup_cosine"
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "43_fafem_frequency_guided_mscb_stage3_warmup_cosine_seed42.ps1"


class FAFEMFrequencyGuidedMSCBTests(unittest.TestCase):
    def test_registered_frequency_guided_variant_has_only_fg_mscb_at_stage3(self):
        config = get_experiment(EXPERIMENT)
        self.assertTrue(config.enable_fafem)
        self.assertTrue(config.enable_fg_mscb_lite_stage3)
        self.assertFalse(config.enable_mscb_lite_stage3)
        self.assertFalse(config.enable_mscb_lite_stage2)
        self.assertFalse(config.enable_mscb_lite_stage1)

        reference = get_experiment(REFERENCE).to_dict()
        candidate = config.to_dict()
        reference.pop("name")
        candidate.pop("name")
        reference.pop("enable_mscb_lite_stage3")
        self.assertEqual(candidate, {
            **reference,
            "enable_fg_mscb_lite_stage3": True,
        })

    def test_fg_mscb_starts_with_uniform_branch_weights_and_preserves_shape(self):
        model = build_experiment_model(
            get_experiment(EXPERIMENT), encoder_weights=None
        ).eval()
        self.assertIsNotNone(model.fg_mscb_lite_stage3)
        with torch.no_grad():
            alpha = model.fg_mscb_lite_stage3.branch_weights(
                torch.randn(2, 1536)
            )
            self.assertTrue(torch.equal(alpha, torch.ones_like(alpha)))
            self.assertTrue(torch.equal(alpha.sum(dim=1), torch.full((2,), 3.0)))
            logits = main_logits(model(torch.randn(1, 3, 352, 352)))
        self.assertEqual(tuple(logits.shape), (1, 1, 352, 352))

    def test_uniform_guidance_reproduces_ordinary_mscb_branch_sum(self):
        model = build_experiment_model(
            get_experiment(EXPERIMENT), encoder_weights=None
        ).eval()
        fg_mscb = model.fg_mscb_lite_stage3
        ordinary_mscb = MSCBLite(384).eval()
        ordinary_mscb.load_state_dict({
            key: value for key, value in fg_mscb.state_dict().items()
            if not key.startswith("guidance_mlp.")
        })
        features = torch.randn(2, 384, 22, 22)
        descriptor = torch.randn(2, 1536)
        with torch.no_grad():
            self.assertTrue(torch.equal(fg_mscb(features, descriptor), ordinary_mscb(features)))

    def test_fg_mscb_is_the_only_mscb_and_is_in_the_decoder_group(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), encoder_weights=None)
        self.assertIsNone(model.mscb_lite_stage3)
        self.assertIsNone(model.mscb_lite_stage2)
        self.assertIsNone(model.mscb_lite_stage1)
        groups = build_optimizer_param_groups(
            model, decoder_lr=3e-4, encoder_lr=4e-5, refine_lr=4.5e-4,
            decoder_weight_decay=1e-4, encoder_weight_decay=5e-2,
            refine_weight_decay=1e-2, profile="layerwise_convnext",
            encoder_layer_decay=0.8,
        )
        decoder_group = next(group for group in groups if group["name"] == "decoder")
        decoder_ids = {id(parameter) for parameter in decoder_group["params"]}
        self.assertEqual(sum(parameter.numel() for parameter in model.fg_mscb_lite_stage3.parameters()), 697_539)
        self.assertTrue({id(parameter) for parameter in model.fafem.parameters()}.issubset(decoder_ids))
        self.assertTrue({id(parameter) for parameter in model.fg_mscb_lite_stage3.parameters()}.issubset(decoder_ids))
        self.assertEqual(decoder_group["lr"], 3e-4)
        self.assertEqual(decoder_group["weight_decay"], 1e-4)
        self.assertEqual(
            sum(isinstance(module, MSCBLite) for module in model.modules()), 1
        )

    def test_fafem_and_frequency_guidance_receive_gradients(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), encoder_weights=None)
        bottleneck = torch.randn(2, 768, 11, 11, requires_grad=True)
        skip = torch.randn(2, 384, 22, 22, requires_grad=True)
        enhanced, descriptor = model.fafem.forward_with_frequency_descriptor(bottleneck)
        output = model.fg_mscb_lite_stage3(skip, descriptor)
        (enhanced.mean() + output.mean()).backward()
        self.assertIsNotNone(model.fafem.low_enhance[0].weight.grad)
        self.assertIsNotNone(model.fg_mscb_lite_stage3.pconv1[0].weight.grad)
        self.assertIsNotNone(model.fg_mscb_lite_stage3.guidance_mlp[0].weight.grad)

    def test_alpha_statistics_use_every_training_sample_with_sample_std(self):
        statistics = summarize_fg_mscb_alpha(torch.tensor([
            [1.0, 1.0, 1.0],
            [0.0, 1.5, 1.5],
            [2.0, 0.5, 0.5],
        ]))
        self.assertEqual(statistics["sample_count"], 3)
        self.assertAlmostEqual(statistics["alpha1_mean"], 1.0)
        self.assertAlmostEqual(statistics["alpha1_std"], 1.0)
        self.assertAlmostEqual(statistics["alpha3_mean"], 1.0)
        self.assertAlmostEqual(statistics["alpha3_std"], 0.5)
        self.assertAlmostEqual(statistics["alpha5_mean"], 1.0)
        self.assertAlmostEqual(statistics["alpha5_std"], 0.5)

    def test_runner_reproduces_experiment_37_protocol_with_fg_mscb_only(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT, check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = next(
            json.loads(line) for line in completed.stdout.splitlines()
            if line.startswith("{")
        )
        arguments = dict(zip(payload["train"][2::2], payload["train"][3::2]))
        self.assertEqual(arguments["--experiment_name"], EXPERIMENT)
        self.assertEqual(arguments["--seed"], "42")
        self.assertEqual(arguments["--enable_fafem"], "True")
        self.assertEqual(arguments["--enable_mscb_lite_stage3"], "False")
        self.assertEqual(arguments["--enable_fg_mscb_lite_stage3"], "True")
        self.assertEqual(arguments["--enable_mscb_lite_stage2"], "False")
        self.assertEqual(arguments["--enable_mscb_lite_stage1"], "False")
        self.assertEqual(arguments["--tta_check"], "False")
        self.assertEqual(arguments["--warmup_epochs"], "5")


if __name__ == "__main__":
    unittest.main()
