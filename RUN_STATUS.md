Status: READY - cumulative FAFEM placement runners implemented; training not started
Seed: 42
Reference: bottleneck FAFEM only (existing behavior unchanged)
Experiment 2: bottleneck + Stage 3
Experiment 3: bottleneck + Stage 3 + Stage 2
Experiment 4: bottleneck + Stage 3 + Stage 2 + Stage 1
Channels: Stage 1=96, Stage 2=192, Stage 3=384, bottleneck=768
Schedule: decoder epochs 1-15; encoder_last_1 at epoch 16
Unfreeze plateau patience: 8 validation epochs
LR plateau patience: 12 validation epochs
Encoder weights: convnext_tiny_22k_1k_384.pth
Batch size: 24; maximum epoch: 350; early-stop patience: 30
Verification: 31 focused tests passed; 352x352 output shape preserved
Artifacts: one_seed_results/ablation/{02_fafem_bottleneck_stage3,03_fafem_bottleneck_stage3_stage2,04_fafem_bottleneck_stage3_stage2_stage1}/seed_42
