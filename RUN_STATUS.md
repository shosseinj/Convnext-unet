Status: READY - Experiment 31 implemented; training not started
Experiment: one_seed_31_fafem_mild_fal
Comparison: FAFEM-only vs FAFEM-only + mild FAL
Seed: 42
Architecture: ConvNeXt-Tiny + decoder + bottleneck FAFEM only
Disabled: uncertainty, CLF, Detail, DySample, MSC, DS, UGBR, gated/FAFEM skips
FAL: probability 0.25; mix <=0.25; region 1-3%
FAL schedule: constant through 60%; anneal to zero at 70%
Protocol: batch 24; plateau scheduler; adaptive unfreeze; maximum 350 epochs
Encoder weights: convnext_tiny_22k_1k_384.pth
Output: one_seed_results/ablation/31_fafem_mild_fal/seed_42
Resume: isolated latest checkpoint with optimizer/scheduler/RNG state
Forward: 1x3x352x352 -> 1x1x352x352; FAFEM-only max diff 0
Verification: 18 focused tests passed; dry-run passed
Runner: ps_one_seed_ablation/31_fafem_mild_fal_seed42.ps1
