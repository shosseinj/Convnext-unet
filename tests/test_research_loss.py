import unittest

import torch

from research_pipeline.losses import DiceBCEBoundaryLoss, supervised_loss


class ResearchLossTests(unittest.TestCase):
    def test_loss_and_gradients_are_finite(self):
        logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
        target = torch.zeros_like(logits); target[:, :, 8:24, 10:22] = 1
        loss = DiceBCEBoundaryLoss()(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_deep_supervision_weights(self):
        target = torch.zeros(1, 1, 8, 8)
        outputs = tuple(torch.zeros_like(target) for _ in range(4))
        criterion = DiceBCEBoundaryLoss()
        expected = criterion(outputs[0], target) * 1.17
        actual = supervised_loss(outputs, target, criterion)
        self.assertAlmostEqual(float(actual), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
