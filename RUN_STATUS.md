Status: LAUNCHING - seed 7777 baseline then FAFEM sequentially
Experiments: one_seed_01_baseline, one_seed_03_baseline_plus_fafem
Seed: 7777
Schedule: decoder epochs 1-15; encoder_last_1 at epoch 16
Unfreeze plateau patience: 8 validation epochs
LR plateau patience: 12 validation epochs
Encoder weights: convnext_tiny_22k_1k_384.pth
Encoder load: verified, missing keys 0
Batch size: 24
Maximum epoch: 350
Early-stop patience: 30
Order: baseline first; FAFEM starts only after baseline runner exits successfully
Artifacts: one_seed_results/ablation/{01_baseline,03_baseline_plus_fafem}/seed_7777
