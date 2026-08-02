import unittest

import torch

from models.architecture_factory import UGBRVariant, build_variant_by_id
from models.ugbr import UGBR


class UGBRTests(unittest.TestCase):
    def test_shapes_identity_uncertainty_and_gradients(self):
        module = UGBR(decoder_channels=12, shallow_channels=7, hidden_channels=8)
        decoder = torch.randn(2, 12, 11, 11, requires_grad=True)
        shallow = torch.randn(2, 7, 22, 22, requires_grad=True)
        initial = torch.randn(2, 1, 44, 44, requires_grad=True)

        output = module(decoder, shallow, initial)

        for key in ("initial_logits", "boundary_logits", "refinement_logits",
                    "final_logits", "uncertainty"):
            self.assertEqual(output[key].shape, initial.shape)
        torch.testing.assert_close(
            output["final_logits"], output["initial_logits"] + output["refinement_logits"],
            rtol=0, atol=0,
        )
        self.assertGreaterEqual(output["uncertainty"].min().item(), 0.0)
        self.assertLessEqual(output["uncertainty"].max().item(), 1.0)

        (output["final_logits"].mean() + output["boundary_logits"].mean()).backward()
        self.assertIsNotNone(decoder.grad)
        self.assertIsNotNone(shallow.grad)
        self.assertIsNotNone(initial.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))

    def test_factory_enables_only_requested_pilot_variants(self):
        ugbr_model = build_variant_by_id("baseline_ugbr", encoder_depth=(1, 1, 1, 1))
        best_ugbr_model = build_variant_by_id(
            "baseline_best_existing_ugbr", encoder_depth=(1, 1, 1, 1)
        )
        baseline_model = build_variant_by_id("baseline", encoder_depth=(1, 1, 1, 1))
        self.assertIsInstance(ugbr_model, UGBRVariant)
        self.assertIsInstance(best_ugbr_model, UGBRVariant)
        self.assertNotIsInstance(baseline_model, UGBRVariant)

    def test_wrapper_exposes_encoder_and_optimizer_groups_are_disjoint(self):
        model = build_variant_by_id("baseline_ugbr", encoder_depth=(1, 1, 1, 1))
        self.assertIs(model.encoder, model.base_model.encoder)

        encoder_parameters = list(model.encoder.parameters())
        # Match the optimizer grouping used by train_research.py.
        decoder_parameters = [parameter for name, parameter in model.named_parameters()
                              if not name.startswith("encoder.")]
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        decoder_ids = {id(parameter) for parameter in decoder_parameters}
        trainable_ids = {id(parameter) for parameter in model.parameters()
                         if parameter.requires_grad}

        self.assertTrue(encoder_ids)
        self.assertTrue(decoder_ids)
        self.assertTrue(encoder_ids.isdisjoint(decoder_ids))
        self.assertEqual(encoder_ids | decoder_ids, trainable_ids)
        self.assertTrue(any(name.startswith("encoder.")
                            for name, _parameter in model.named_parameters()))
        self.assertFalse(any(name.startswith("base_model.encoder.")
                             for name, _parameter in model.named_parameters()))
        self.assertFalse(any(key.startswith("encoder.") for key in model.state_dict()))
        self.assertTrue(any(key.startswith("base_model.encoder.") for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
