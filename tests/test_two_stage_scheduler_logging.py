from types import SimpleNamespace

import torch

from main_torch import TwoStageWarmupCosineRestart, describe_scheduler


def test_two_stage_scheduler_has_a_safe_log_description():
    optimizer = torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.ones(()))], "lr": 3e-4}])
    scheduler = TwoStageWarmupCosineRestart(optimizer, 5, 120, 200, 1e-6, 0.1, 1 / 3)
    args = SimpleNamespace(
        cosine_t0=8, cosine_t_mult=2, min_lr=1e-6,
        warmup_epochs=5, restart_epoch=120,
        first_cycle_min_factor=0.1, restart_factor=1 / 3,
    )
    assert describe_scheduler(args, "two_stage_warmup_cosine_restart", scheduler) == (
        "warmup_epochs=5,restart_epoch=120,first_cycle_min_factor=0.1,"
        "restart_factor=0.3333333333333333,eta_min=1.00e-06)"
    )
