import torch
import pytest

from main_torch import TwoStageWarmupCosineRestart


def _optimizer():
    return torch.optim.AdamW([
        {"params": [torch.nn.Parameter(torch.ones(()))], "lr": 3e-4},
        {"params": [torch.nn.Parameter(torch.ones(()))], "lr": 3e-5},
    ])


def test_two_stage_scheduler_warms_up_restarts_once_and_reaches_minimum_lr():
    optimizer = _optimizer()
    scheduler = TwoStageWarmupCosineRestart(
        optimizer, warmup_epochs=5, restart_epoch=120, total_epochs=200,
        min_lr=1e-6, first_cycle_min_factor=0.1, restart_factor=1 / 3,
    )
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([3e-5, 3e-6])

    for _ in range(4):
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([3e-4, 3e-5])

    for _ in range(116):
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-4, 1e-5])

    for _ in range(80):
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-6, 1e-6])


def test_two_stage_scheduler_resume_preserves_position_and_lrs():
    optimizer = _optimizer()
    scheduler = TwoStageWarmupCosineRestart(optimizer, 5, 120, 200, 1e-6, 0.1, 1 / 3)
    for _ in range(137):
        scheduler.step()
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = scheduler.state_dict()
    expected = [group["lr"] for group in optimizer.param_groups]

    resumed_optimizer = _optimizer()
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler = TwoStageWarmupCosineRestart(resumed_optimizer, 5, 120, 200, 1e-6, 0.1, 1 / 3)
    resumed_scheduler.load_state_dict(saved_scheduler)
    assert [group["lr"] for group in resumed_optimizer.param_groups] == pytest.approx(expected)

    scheduler.step()
    resumed_scheduler.step()
    assert [group["lr"] for group in resumed_optimizer.param_groups] == pytest.approx([group["lr"] for group in optimizer.param_groups])
