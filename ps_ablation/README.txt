# Polyp Segmentation Ablation Launchers

Three seeds are used in every experiment:
- 42
- 6543
- 7777

Results are stored as:

results/
└── ablation/
    ├── 01_baseline/
    │   ├── seed_42/
    │   ├── seed_6543/
    │   └── seed_7777/
    ├── 02_add_msc/
    └── ...

Important:
The PowerShell files assume the training program accepts these arguments:

--seed
--output_dir
--use_msc
--use_lrse
--use_detail_branch
--detail_fusion
--deep_supervision

If your existing train.py uses different flag names, change only those argument names.
Do not change the experiment definitions.

Experiment 11 (full without DS) is identical to experiment 05 (+GDF without DS).
You therefore do NOT need to train it twice; reuse experiment 05 results in the mechanism-isolation table.
