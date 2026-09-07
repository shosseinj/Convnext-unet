Status: PHASE 5 RUNTIME BENCHMARK COMPLETE
Hardware: NVIDIA GeForce RTX 4090; PyTorch 2.5.1+cu124; CUDA 12.4
Protocol: FP16 autocast, batch 1, input 1x3x352x352, forward pass only, no TTA
Timing: 3 repetitions; each has 100 warm-ups and 500 synchronized CUDA-event iterations
Aggregation: mean +/- sample SD of the three repetition means
Checkpoints: seed 42; strict metadata, architecture, and state-dict validation passed for all four models
Output validation: every model returned 1x1x352x352
Complexity: project THOP convention, FLOPs = 2 x MACs
Result: Baseline 10.573 +/- 0.196 ms, 94.585 FPS, 248.859 MiB
Result: FAFEM 11.014 +/- 0.137 ms, 90.794 FPS, 249.832 MiB
Result: FAFEM + MSCB 11.630 +/- 0.137 ms, 85.986 FPS, 252.876 MiB
Result: Residual RFG-MSCB 11.806 +/- 0.023 ms, 84.701 FPS, 253.306 MiB
Artifacts: benchmark_runtime.py, runtime_benchmark_results.csv, runtime_benchmark_results.json
Training/manuscript: no retraining; no model, weight, manuscript, or DOCX changes
