Status: READY - FAFEM + Cross-Level Fusion implemented; new training not started
Experiment: one_seed_05_fafem_plus_cross_level_fusion
Seeds: 42, 7777, 6543
Reference: bottleneck FAFEM only; existing behavior and checkpoints unchanged
Skip inputs: f1=96x88x88, f2=192x44x44, f3=384x22x22 at input 352
Bottleneck: f4=768x11x11 with existing FAFEM
Cross-Level Fusion: 133,890 parameters; decoder LR group from epoch 1
Total: 29,493,765 parameters
Schedule: decoder epochs 1-15; adaptive encoder unfreeze unchanged
Encoder weights: convnext_tiny_22k_1k_384.pth
Batch size: 24; maximum epoch: 350; early-stop patience: 30
Verification: 38 focused tests passed; output remains 1x1x352x352
Runner: ps_one_seed_ablation/05_fafem_plus_cross_level_fusion.ps1
Artifacts: one_seed_results/ablation/05_fafem_plus_cross_level_fusion/seed_{42,7777,6543}
