"""Reproducible forward-pass benchmark for the final internal ablations."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import statistics
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

from ablation_registry import get_experiment
from checkpoint_management import load_checkpoint_file, strip_thop_state
from evaluation_core import count_parameters, measure_complexity
from one_seed_models import build_experiment_model, main_logits


MODELS = (
    ("Baseline", "one_seed_01_baseline", "01_baseline"),
    ("FAFEM", "one_seed_03_baseline_plus_fafem", "03_baseline_plus_fafem"),
    ("FAFEM + MSCB", "one_seed_37_fafem_mscb_lite_stage3_warmup_cosine", "37_fafem_mscb_lite_stage3_warmup_cosine"),
    ("Residual RFG-MSCB (Ours)", "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine", "45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--input-size", type=int, default=352)
    parser.add_argument("--encoder-weights", type=Path, default=Path("convnext_tiny_22k_1k_384.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("runtime_benchmark_results.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("runtime_benchmark_results.csv"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_context(precision: str):
    return torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()


def forward(model, sample, precision):
    with torch.inference_mode(), inference_context(precision):
        return main_logits(model(sample))


def load_verified_model(spec, args, device):
    label, experiment_name, result_dir = spec
    config = get_experiment(experiment_name)
    checkpoint_path = Path("one_seed_results/ablation") / result_dir / f"seed_{args.seed}" / "best_checkpoint.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = load_checkpoint_file(checkpoint_path)
    if checkpoint.get("training_complete") is not True:
        raise RuntimeError(f"Incomplete checkpoint: {checkpoint_path}")
    if checkpoint.get("experiment_name") != config.name or checkpoint.get("seed") != args.seed:
        raise RuntimeError(f"Experiment/seed mismatch: {checkpoint_path}")
    if checkpoint.get("architecture") != config.to_dict():
        raise RuntimeError(f"Architecture metadata mismatch: {checkpoint_path}")
    model = build_experiment_model(config, args.encoder_weights, device)
    model.load_state_dict(strip_thop_state(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    return label, config, checkpoint_path.resolve(), model


def benchmark_once(model, sample, precision, warmup, iterations):
    for _ in range(warmup):
        output = forward(model, sample, precision)
    torch.cuda.synchronize()
    if tuple(output.shape) != (1, 1, sample.shape[-2], sample.shape[-1]):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = forward(model, sample, precision)
        end.record()
        end.synchronize()
        latencies.append(float(start.elapsed_time(end)))
    torch.cuda.synchronize()
    return {
        "mean_latency_ms": statistics.fmean(latencies),
        "iteration_sd_ms": statistics.stdev(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "output_shape": list(output.shape),
    }


def relative_change(old, new, key):
    delta = new[key] - old[key]
    return {"delta": delta, "percent": 100.0 * delta / old[key]}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup < 100 or args.iterations < 500 or args.repetitions < 3:
        raise ValueError("Protocol requires >=100 warmups, >=500 iterations, and >=3 repetitions")

    device = torch.device("cuda:0")
    sample = torch.zeros(1, 3, args.input_size, args.input_size, device=device)
    results = []
    for spec in MODELS:
        label, config, checkpoint_path, model = load_verified_model(spec, args, device)
        trainable, total = count_parameters(model)
        complexity = measure_complexity(model, args.input_size)
        results.append({
            "model": label,
            "experiment_name": config.name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "parameters": total,
            "parameters_m": total / 1e6,
            "trainable_parameters": trainable,
            **complexity,
            "repetition_results": [],
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Rotate the starting model on every repetition to reduce order bias.
    for repetition in range(args.repetitions):
        order = list(range(len(MODELS)))
        order = order[repetition % len(order):] + order[:repetition % len(order)]
        for index in order:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _, _, _, model = load_verified_model(MODELS[index], args, device)
            run = benchmark_once(model, sample, args.precision, args.warmup, args.iterations)
            run["repetition"] = repetition + 1
            run["order_position"] = order.index(index) + 1
            results[index]["repetition_results"].append(run)
            print(f"rep={repetition + 1} model={results[index]['model']} mean_ms={run['mean_latency_ms']:.4f}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    for result in results:
        runs = result["repetition_results"]
        means = [run["mean_latency_ms"] for run in runs]
        result["latency_mean_ms"] = statistics.fmean(means)
        result["latency_sample_sd_ms"] = statistics.stdev(means)
        result["median_latency_ms"] = statistics.fmean(run["median_latency_ms"] for run in runs)
        result["p95_latency_ms"] = statistics.fmean(run["p95_latency_ms"] for run in runs)
        result["fps"] = 1000.0 / result["latency_mean_ms"]
        result["aggregate_fps"] = args.iterations * args.repetitions / (sum(means) * args.iterations / 1000.0)
        result["peak_vram_mib"] = max(run["peak_vram_mib"] for run in runs)
        result["output_shape"] = runs[0]["output_shape"]

    overhead = []
    keys = ("parameters", "gflops", "latency_mean_ms", "fps", "peak_vram_mib")
    for old, new in zip(results, results[1:]):
        overhead.append({
            "transition": f"{old['model']} -> {new['model']}",
            **{key: relative_change(old, new, key) for key in keys},
        })

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
            "precision": args.precision,
            "batch_size": 1,
            "input_shape": [1, 3, args.input_size, args.input_size],
        },
        "protocol": {
            "scope": "model forward pass only; no data loading, disk I/O, post-processing, or TTA",
            "timing": "per-iteration torch.cuda.Event with end-event synchronization",
            "warmup_iterations_per_repetition": args.warmup,
            "measured_iterations_per_repetition": args.iterations,
            "repetitions": args.repetitions,
            "aggregation": "mean and sample SD of repetition means",
            "peak_memory": "maximum torch.cuda.max_memory_allocated after post-warmup reset; model + input + activations",
            "complexity": "project evaluation_core.measure_complexity; THOP MACs and FLOPs = 2 x MACs",
        },
        "results": results,
        "relative_overhead": overhead,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = ("model", "experiment_name", "checkpoint", "parameters", "parameters_m", "gmacs", "gflops", "latency_mean_ms", "latency_sample_sd_ms", "median_latency_ms", "p95_latency_ms", "fps", "aggregate_fps", "peak_vram_mib", "output_shape")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: json.dumps(result[key]) if key == "output_shape" else result[key] for key in fields})
    print(f"JSON: {args.output_json.resolve()}")
    print(f"CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
