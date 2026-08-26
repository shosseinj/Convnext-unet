Status: READY - multi-resolution CLF-v2 implemented; CLF-v2 training not started
Experiment: one_seed_06_fafem_plus_cross_level_fusion_v2
Seeds: 42, 7777, 6543
References: bottleneck FAFEM and CLF-v1 unchanged
Skip inputs: f1=96x88x88, f2=192x44x44, f3=384x22x22 at input 352
Bottleneck: f4=768x11x11 with existing FAFEM
Contexts: 88 uses f1+f2; 44 uses f1+f2+f3; 22 uses f2+f3
CLF-v2: 93,156 parameters; 1e-3 LayerScale; decoder LR from epoch 1
Total: 29,453,031 parameters
Schedule: decoder epochs 1-15; adaptive encoder unfreeze unchanged
Encoder weights: convnext_tiny_22k_1k_384.pth
Batch size: 24; maximum epoch: 350; early-stop patience: 30
Verification: 47 focused tests passed; output remains 1x1x352x352
Runner: ps_one_seed_ablation/06_fafem_plus_cross_level_fusion_v2.ps1
Artifacts: one_seed_results/ablation/06_fafem_plus_cross_level_fusion_v2/seed_{42,7777,6543}
