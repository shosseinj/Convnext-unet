import unittest

import torch
import torch.nn as nn

from train_research import EVIDENCE_SOURCE_FILES, ROOT, load_runner_variants
from research_pipeline.losses import (
    morphological_gradient_target,
    supervised_loss,
    ugbr_composite_loss,
)


class UGBRLossTests(unittest.TestCase):
    @staticmethod
    def outputs(initial, final, boundary, uncertainty):
        return {
            "initial_logits": initial,
            "final_logits": final,
            "boundary_logits": boundary,
            "uncertainty": uncertainty,
        }

    def test_formula_and_components_are_exposed(self):
        target = torch.zeros(1, 1, 5, 5)
        initial = torch.full_like(target, -1.0)
        final = torch.full_like(target, 0.5)
        boundary = torch.zeros_like(target)
        uncertainty = torch.zeros_like(target)
        criterion = nn.BCEWithLogitsLoss()
        parts = ugbr_composite_loss(
            self.outputs(initial, final, boundary, uncertainty), target, criterion
        )
        expected = (parts["seg_final"] + 0.4 * parts["seg_initial"] +
                    0.2 * parts["boundary"] + 0.1 * parts["consistency"])
        self.assertEqual(set(parts), {"total", "seg_final", "seg_initial", "boundary", "consistency"})
        torch.testing.assert_close(parts["total"], expected)

    def test_morphological_gradient_target(self):
        target = torch.zeros(1, 1, 7, 7)
        target[:, :, 2:5, 2:5] = 1
        expected = torch.zeros_like(target)
        expected[:, :, 1:6, 1:6] = 1
        expected[:, :, 3, 3] = 0
        torch.testing.assert_close(morphological_gradient_target(target), expected)

    def test_consistency_masks_uncertain_pixels(self):
        target = torch.zeros(1, 1, 1, 2)
        initial = torch.tensor([[[[-4.0, 0.0]]]])
        final = torch.tensor([[[[4.0, 8.0]]]])
        boundary = torch.zeros_like(target)
        uncertainty = 1.0 - torch.abs(2.0 * torch.sigmoid(initial) - 1.0)
        parts = ugbr_composite_loss(
            self.outputs(initial, final, boundary, uncertainty), target, nn.BCEWithLogitsLoss()
        )
        expected = (torch.sigmoid(final[..., 0]) - torch.sigmoid(initial[..., 0])).square().mean()
        torch.testing.assert_close(parts["consistency"], expected)

    def test_finite_gradients(self):
        target = torch.zeros(2, 1, 8, 8)
        target[:, :, 2:6, 2:6] = 1
        initial = torch.randn_like(target, requires_grad=True)
        refinement = torch.randn_like(target, requires_grad=True)
        boundary = torch.randn_like(target, requires_grad=True)
        final = initial + refinement
        uncertainty = 1.0 - torch.abs(2.0 * torch.sigmoid(initial) - 1.0)
        parts = ugbr_composite_loss(
            self.outputs(initial, final, boundary, uncertainty), target, nn.BCEWithLogitsLoss()
        )
        parts["total"].backward()
        for tensor in (initial, refinement, boundary):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_legacy_deep_supervision_compatibility(self):
        target = torch.zeros(1, 1, 4, 4)
        outputs = tuple(torch.zeros_like(target) for _ in range(4))
        criterion = nn.BCEWithLogitsLoss()
        expected = criterion(outputs[0], target) * 1.17
        torch.testing.assert_close(supervised_loss(outputs, target, criterion), expected)

    def test_runner_resolves_complete_pilot_matrix_without_training(self):
        variants = load_runner_variants(ROOT / "configs" / "ablation_matrix.yaml")
        pilot_ids = {
            "baseline",
            "baseline_ugbr",
            "baseline_best_existing",
            "baseline_best_existing_ugbr",
        }
        self.assertTrue(pilot_ids.issubset(variants))
        self.assertFalse(variants["baseline"].ugbr)
        self.assertTrue(variants["baseline_ugbr"].ugbr)
        self.assertFalse(variants["baseline_best_existing"].ugbr)
        self.assertTrue(variants["baseline_best_existing_ugbr"].ugbr)

    def test_ugbr_source_is_fingerprinted(self):
        self.assertIn("models/ugbr.py", EVIDENCE_SOURCE_FILES)


if __name__ == "__main__":
    unittest.main()
