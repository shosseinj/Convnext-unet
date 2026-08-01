## Limitations

The persisted development manifests define deterministic image-level training and validation partitions. Patient, procedure, and source-video identifiers were not available in the audited dataset layout; consequently, patient-level or video-level separation and duplicate screening across those identifiers could not be verified. This limitation must be considered when interpreting internal validation results.

The experimental protocol uses three prespecified random seeds. Results are therefore reported with per-seed values, the sample standard deviation, and a two-sided 95% Student-t interval with two degrees of freedom. Given the small number of seeds, these intervals may be wide, and descriptive differences must not be presented as evidence of statistical significance unless a separately prespecified and validated test supports that conclusion.

Checkpoint selection uses only the equal-weight mean validation Dice of Kvasir-SEG and CVC-ClinicDB. Evaluation uses a fixed threshold of 0.5 without test-time augmentation. The three external datasets are reserved for evaluation and are not used for checkpoint selection or tuning. S-measure is not included because it is not implemented in the validated evaluation pipeline.

Parameters and operation counts are reported from the audited 352 by 352 architecture gate, using FLOPs equal to twice the measured MAC count. Latency, throughput, and peak-memory values are omitted because no reproducible hardware benchmark satisfying the evidence contract has been completed. These omissions should not be replaced with estimates.
