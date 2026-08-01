#!/usr/bin/env python3
"""Gate 1 validation for every incremental architecture variant."""

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from thop import profile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.architecture_factory import build_variant, load_variant_configs


def output_shapes(output):
    tensors = output if isinstance(output, (tuple, list)) else (output,)
    return [list(tensor.shape) for tensor in tensors]


def pretrained_path(config):
    paths = {
        "resnet34": ROOT / "pretrained" / "resnet34-b627a593.pth",
        "efficientnet_b0": ROOT / "pretrained" / "efficientnet_b0_rwightman-7f5810bc.pth",
    }
    return paths.get(config.backbone)


def validate_variant(config, input_size, device):
    weights_path = pretrained_path(config)
    model = build_variant(config, weights_path=weights_path, drop_path_rate=0.0, dropout_rate=0.0).to(device)
    model.eval()
    sample = torch.zeros(1, 3, input_size, input_size, device=device)
    with torch.inference_mode():
        output = model(sample)
    shapes = output_shapes(output)
    expected_outputs = 1 + config.deep_supervision_heads
    expected_shape = [1, 1, input_size, input_size]
    if len(shapes) != expected_outputs or any(shape != expected_shape for shape in shapes):
        raise AssertionError(f"{config.id}: unexpected output shapes {shapes}")

    state = model.state_dict()
    clone = build_variant(config, weights_path=weights_path, drop_path_rate=0.0, dropout_rate=0.0).to(device)
    clone.load_state_dict(state, strict=True)
    incompatible_error = False
    damaged = dict(state)
    damaged.pop(next(iter(damaged)))
    try:
        clone.load_state_dict(damaged, strict=True)
    except RuntimeError as exc:
        incompatible_error = "Missing key(s)" in str(exc)
    if not incompatible_error:
        raise AssertionError(f"{config.id}: incompatible checkpoint did not fail clearly")

    macs, _ = profile(model, inputs=(sample,), verbose=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "variant": config.id,
        "status": "PASS",
        "output_shapes": shapes,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "macs": int(macs),
        "gflops_convention_2x_macs": float(2 * macs / 1e9),
        "strict_round_trip": True,
        "incompatible_checkpoint_error": True,
    }
    del model, clone, state, sample, output
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs" / "ablation_matrix.yaml")
    parser.add_argument("--input-size", type=int, default=352)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sections", nargs="+", default=["incremental"],
                        choices=("incremental", "controls"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "architecture_gate.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    variants = load_variant_configs(args.matrix, sections=tuple(args.sections))
    results = []
    for config in variants.values():
        print(f"Validating {config.id}...", flush=True)
        result = validate_variant(config, args.input_size, device)
        results.append(result)
        print(f"  PASS params={result['parameters']:,} MACs={result['macs']:,}", flush=True)
    report = {
        "gate": "architecture",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_size": [args.input_size, args.input_size],
        "device": str(device),
        "matrix_sections": args.sections,
        "flops_definition": "2 * MACs",
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Gate 1 PASS: {args.output}")


if __name__ == "__main__":
    main()
