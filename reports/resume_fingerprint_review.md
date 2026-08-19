# Run identity
Variant: baseline_msc_bsei_detail_gdf; seed: 2026; checkpoint epoch: 27; git commit: checkpoint 740454776a9c82974fb29056741429de9890f016, current 5cbf60e9a6ee04dff1d431449fdf4a00f483fa0f.

# Checkpoint path
`results/raw/baseline_msc_bsei_detail_gdf/seed_2026/official/last.pth`

# Mismatched fingerprints
- `train_research.py`: checkpoint `4754aa606cf0fa168dce9837645cc2f7455bb0712c42eb968799a2f996b06349` / 16888 bytes; current `6925cc2a8c5d1e5b6440bef56e333404aa518811d7b94b87ca5744eb0bf5c456` / 18123 bytes.
- `models/architecture_factory.py`: checkpoint `97ce628cd116e8be50d11a4d0d87e33fbef9f2d9c6695dd93a9dbc5e49b3053a` / 3780 bytes; current `40050d78c368c1e9946c74cc310b3e816903bb87df9dc0f1399f4a565274f9ab` / 6523 bytes.
- `research_pipeline/losses.py`: checkpoint `fecc6f5ee6b03ae6a88e037bb8b7f74ede5e6ebda349436621bad8df728af0fc` / 2135 bytes; current `59dab671f570a17dc0e348ea9fa56bfa95b99e38d25551f8944469ce26fb4040` / 5245 bytes.
- `configs/ablation_matrix.yaml`: checkpoint `8ed4b32dc20b07aa56845e26649fcfa57b5b5e1652598eaf8c4d39e1bae5d98d` / 3623 bytes; current `04f9b25017f9fca996815736e6421882e35a8db6f54bc6f29f24ed24a273ef2b` / 4000 bytes.

# Changed files
`train_research.py`, `models/architecture_factory.py`, `research_pipeline/losses.py`, and `configs/ablation_matrix.yaml` changed. Matching inputs: `models/convnext_pretrain.py`, `research_pipeline/data.py`, `research_pipeline/reproducibility.py`, `configs/training_protocol.yaml`, `configs/splits/development_seed_2026.json`, and pretrained weights `convnext_tiny_22k_1k_384.pth` (`f22f8850dcded245f4ae3bc402bbde9c6f5e4e23e76dfaea3aadb75aa9d3705a`, 114414741 bytes). Dataset identity is Kvasir-SEG/CVC-ClinicDB development with the same split hash `0b0156afeafd7f12f696449c9259ea6be9f5cff803cec9131253d36cb64e96a6`; model parameters are 29,601,107; loss identity is `dice_bce_boundary`; training hyperparameters match the stored protocol.

# Classification
All four changed files are Training-affecting: runner/recovery semantics, model architecture, loss implementation, and experiment configuration. No workflow-only or logging-only mismatch was found.

# Resume decision
RESUME DENIED. The old partial run is non-resumable under immutable validation. Its metadata and checkpoints remain unchanged.

# Required next action
Use the fresh isolated namespace `results/raw_clean_recovery_20260802/baseline_msc_bsei_detail_gdf/seed_2026/official` for a clean epoch-0 run. Do not launch until the approved launcher is explicitly configured for this namespace.
