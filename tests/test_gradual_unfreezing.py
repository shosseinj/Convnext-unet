import unittest

import torch

from main_torch import (
    advance_encoder_unfreezing, create_plateau_scheduler, encoder_stage_name,
    grad_clip_norm_for_epoch, set_frozen_modules_eval, set_training_stage,
)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample_layers = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1) for _ in range(4)]
        )
        self.stages = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1) for _ in range(4)]
        )
        self.dropout = torch.nn.Dropout(0.25)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.decoder = torch.nn.Linear(1, 1)


class GradualEncoderUnfreezingTests(unittest.TestCase):
    def test_gradient_clipping_is_relaxed_only_during_frozen_encoder_epochs(self):
        self.assertEqual(grad_clip_norm_for_epoch(0, 15), 5.0)
        self.assertEqual(grad_clip_norm_for_epoch(14, 15), 5.0)
        self.assertEqual(grad_clip_norm_for_epoch(15, 15), 3.5)

    def test_plateau_unfreezes_one_encoder_stage_after_eight_epochs(self):
        depth = 0
        plateau = 0
        for _ in range(7):
            depth, plateau, changed = advance_encoder_unfreezing(
                depth, plateau, improved=False, patience=8
            )
            self.assertFalse(changed)
        depth, plateau, changed = advance_encoder_unfreezing(
            depth, plateau, improved=False, patience=8
        )
        self.assertEqual((depth, plateau, changed), (1, 0, True))
        self.assertEqual(encoder_stage_name(depth), "encoder_last_1")

    def test_improvement_resets_plateau_counter(self):
        self.assertEqual(
            advance_encoder_unfreezing(
                2, 7, improved=True, patience=8
            ),
            (2, 0, False),
        )

    def test_five_plateaus_eventually_unfreeze_all_encoder_sections(self):
        depth = 0
        plateau = 0
        transitions = []
        for _ in range(40):
            depth, plateau, changed = advance_encoder_unfreezing(
                depth, plateau, improved=False, patience=8
            )
            if changed:
                transitions.append(encoder_stage_name(depth))
        self.assertEqual(
            transitions,
            [
                "encoder_last_1", "encoder_last_2", "encoder_last_3",
                "encoder_last_4", "all",
            ],
        )

    def test_four_stages_can_train_while_stem_remains_frozen(self):
        model = TinyModel()
        set_training_stage(model, "encoder_last_4")

        self.assertTrue(
            all(parameter.requires_grad for parameter in model.encoder.stages.parameters())
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.encoder.downsample_layers[0].parameters()
            )
        )
        for index in range(1, 4):
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for parameter in model.encoder.downsample_layers[index].parameters()
                )
            )

    def test_plateau_scheduler_monitors_iou_with_patience_twelve(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = create_plateau_scheduler(
            optimizer, min_lr=1e-6, patience=12
        )

        self.assertEqual(scheduler.mode, "max")
        self.assertEqual(scheduler.patience, 12)
        self.assertEqual(scheduler.factor, 0.9)
        scheduler.step(0.8)
        for _ in range(13):
            scheduler.step(0.7)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 9e-5)

    def test_partial_stage_keeps_earlier_encoder_stages_frozen(self):
        model = TinyModel()
        set_training_stage(model, "encoder_last_2")

        for index in range(4):
            expected = index >= 2
            self.assertEqual(
                all(parameter.requires_grad for parameter in model.encoder.stages[index].parameters()),
                expected,
            )
            self.assertEqual(
                all(parameter.requires_grad for parameter in model.encoder.downsample_layers[index].parameters()),
                expected,
            )
        self.assertTrue(all(parameter.requires_grad for parameter in model.decoder.parameters()))

    def test_partial_unfreezing_keeps_shared_encoder_dropout_disabled(self):
        model = TinyModel()
        set_training_stage(model, "encoder_last_1")
        model.train()
        set_frozen_modules_eval(model)

        self.assertFalse(model.encoder.stages[0].training)
        self.assertTrue(model.encoder.stages[3].training)
        self.assertFalse(model.encoder.dropout.training)


if __name__ == "__main__":
    unittest.main()
