# ABLATION_SOURCE_MANIFEST.md

Source provenance for every experiment in the ablation evidence package.
Each entry lists the exact files that back the reported metrics.

## Exp33 — one_seed_33_fafem_warmup_cosine

| Seed | evaluation_summary.json | best_checkpoint.pth |
|------|------------------------|---------------------|
| 42   | one_seed_results/ablation/33_fafem_warmup_cosine/seed_42/evaluation_summary.json | one_seed_results/ablation/33_fafem_warmup_cosine/seed_42/best_checkpoint.pth |
| 6543 | one_seed_results/ablation/33_fafem_warmup_cosine/seed_6543/evaluation_summary.json | one_seed_results/ablation/33_fafem_warmup_cosine/seed_6543/best_checkpoint.pth |
| 7777 | one_seed_results/ablation/33_fafem_warmup_cosine/seed_7777/evaluation_summary.json | one_seed_results/ablation/33_fafem_warmup_cosine/seed_7777/best_checkpoint.pth |

Config source: ablation_registry.py (registry entry `one_seed_33_fafem_warmup_cosine`)
Runner: main_torch.py via ablation_cli.py

## Exp37 — one_seed_37_fafem_mscb_lite_stage3_warmup_cosine

| Seed | evaluation_summary.json | best_checkpoint.pth |
|------|------------------------|---------------------|
| 42   | one_seed_results/ablation/37_fafem_mscb_lite_stage3_warmup_cosine/seed_42/evaluation_summary.json | one_seed_results/ablation/37_fafem_mscb_lite_stage3_warmup_cosine/seed_42/best_checkpoint.pth |
| 6543 | one_seed_results/ablation/37_fafem_mscb_lite_stage3_warmup_cosine/seed_6543/evaluation_summary.json | one_seed_results/ablation/37_fafem_mscb_lite_stage3_warmup_cosine/seed_6543/best_checkpoint.pth |
| 7777 | one_seed_results/ablation/37_fafem_mscb_lite_stage3_warmup_cosine/seed_7777/evaluation_summary.json | one_seed_results/ablation/37_fafem_mscb_lite_stage3_warmup_cosine/seed_7777/best_checkpoint.pth |

Config source: ablation_registry.py (registry entry `one_seed_37_fafem_mscb_lite_stage3_warmup_cosine`)
Runner: main_torch.py via ablation_cli.py

## Exp41 — one_seed_41_fafem_mscb_lite_stage3_stage2_warmup_cosine

| Seed | evaluation_summary.json | best_checkpoint.pth |
|------|------------------------|---------------------|
| 42   | one_seed_results/ablation/41_fafem_mscb_lite_stage3_stage2_warmup_cosine/seed_42/evaluation_summary.json | one_seed_results/ablation/41_fafem_mscb_lite_stage3_stage2_warmup_cosine/seed_42/best_checkpoint.pth |

Config source: ablation_registry.py (registry entry `one_seed_41_fafem_mscb_lite_stage3_stage2_warmup_cosine`)
Runner: main_torch.py via ablation_cli.py

## Exp42 — one_seed_42_fafem_mscb_lite_stage3_stage2_stage1_warmup_cosine

| Seed | evaluation_summary.json | best_checkpoint.pth |
|------|------------------------|---------------------|
| 42   | one_seed_results/ablation/42_fafem_mscb_lite_stage3_stage2_stage1_warmup_cosine/seed_42/evaluation_summary.json | one_seed_results/ablation/42_fafem_mscb_lite_stage3_stage2_stage1_warmup_cosine/seed_42/best_checkpoint.pth |

Config source: ablation_registry.py (registry entry `one_seed_42_fafem_mscb_lite_stage3_stage2_stage1_warmup_cosine`)
Runner: main_torch.py via ablation_cli.py

## Exp43 — one_seed_43_fafem_frequency_guided_mscb_stage3_warmup_cosine

| Seed | evaluation_summary.json | best_checkpoint.pth |
|------|------------------------|---------------------|
| 42   | one_seed_results/ablation/43_fafem_frequency_guided_mscb_stage3_warmup_cosine/seed_42/evaluation_summary.json | one_seed_results/ablation/43_fafem_frequency_guided_mscb_stage3_warmup_cosine/seed_42/best_checkpoint.pth |

Config source: ablation_registry.py (registry entry `one_seed_43_fafem_frequency_guided_mscb_stage3_warmup_cosine`)
Runner: main_torch.py via ablation_cli.py

## Exp45 — one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine

| Seed | evaluation_summary.json | best_checkpoint.pth |
|------|------------------------|---------------------|
| 42   | one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_42/evaluation_summary.json | one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_42/best_checkpoint.pth |
| 6543 | one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_6543/evaluation_summary.json | one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_6543/best_checkpoint.pth |
| 7777 | one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_7777/evaluation_summary.json | one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_7777/best_checkpoint.pth |

Config source: ablation_registry.py (registry entry `one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine`)
Runner: main_torch.py via ablation_cli.py

Snapshots in this package:
- source_snapshots/exp45/seed_42_evaluation_summary.json
- source_snapshots/exp45/seed_6543_evaluation_summary.json
- source_snapshots/exp45/seed_7777_evaluation_summary.json

## Shared reference files

- source_snapshots/ablation_registry.py — canonical experiment definitions
- source_snapshots/evaluation_core.py — evaluation metric implementation
- source_snapshots/checkpoint_metadata_seed42.json — architecture metadata for seed 42
