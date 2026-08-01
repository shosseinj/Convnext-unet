#!/usr/bin/env python3
"""Run ConvNeXt-UNet segmentation on every video frame."""

import argparse
import csv
import importlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from video_evaluation import (annotate_prediction, binary_metrics, filter_components,
                              logits_to_probability, preprocess_frame, resize_probability)


METRIC_NAMES = ("dice", "iou", "precision", "recall", "specificity", "pixel_accuracy")


def build_model(spec, encoder_weights, device):
    try:
        module_name, class_name = spec.split(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise RuntimeError(f"Invalid --model '{spec}': {exc}") from exc
    weights = str(encoder_weights) if encoder_weights else None
    return cls(weights_path=weights, num_classes=1, encoder_depth=[3, 3, 9, 3]).to(device)


def load_checkpoint_strict(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must contain a state dictionary")
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    state = {}
    ignored_profile_tensors = 0
    for key, value in checkpoint.items():
        key = key[7:] if key.startswith("module.") else key
        if key.endswith("total_ops") or key.endswith("total_params"):
            ignored_profile_tensors += 1
            continue
        if torch.is_tensor(value):
            state[key] = value
    if not state:
        raise RuntimeError("No tensor state dictionary found in checkpoint")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint weights do not match model architecture:\n{exc}") from exc
    print(f"Loaded {len(state)} tensors from {path}"
          f" (ignored {ignored_profile_tensors} profiling tensors)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-video", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-video", required=True, type=Path)
    p.add_argument("--model", default="models.convnext_pretrain:ConvNeXtUNet")
    p.add_argument("--encoder-weights", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--input-size", type=int, default=352)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=100)
    p.add_argument("--mask-alpha", type=float, default=0.35)
    p.add_argument("--contour-thickness", type=int, default=2)
    p.add_argument("--box-thickness", type=int, default=2)
    p.add_argument("--output-fps", type=float)
    p.add_argument("--codec", default="mp4v")
    p.add_argument("--display", action="store_true")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--ground-truth-dir", type=Path)
    p.add_argument("--show-ground-truth", action="store_true")
    p.add_argument("--side-by-side", action="store_true")
    p.add_argument("--metrics-csv", type=Path)
    p.add_argument("--metrics-json", type=Path)
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def read_manifest(path):
    if not path:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("frames", data)
    return {int(item["frame_index"]): item for item in entries}


def read_gt(args, entry, shape):
    if not args.ground_truth_dir or not entry:
        return None
    name = entry.get("ground_truth_mask_filename") or entry.get("source_image_filename")
    if not name:
        return None
    path = args.ground_truth_dir / name
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    height, width = shape[:2]
    if entry.get("letterbox"):
        scale = min(width / mask.shape[1], height / mask.shape[0])
        resized = cv2.resize(mask, (round(mask.shape[1] * scale), round(mask.shape[0] * scale)),
                             interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((height, width), dtype=np.uint8)
        y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        mask = canvas
    else:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def main():
    args = parse_args()
    for path, label in ((args.input_video, "input video"), (args.checkpoint, "checkpoint")):
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")
    if args.ground_truth_dir and not args.ground_truth_dir.is_dir():
        raise SystemExit(f"Ground-truth directory does not exist: {args.ground_truth_dir}")
    if not 0 <= args.threshold <= 1 or not 0 <= args.mask_alpha <= 1 or args.min_area < 0:
        raise SystemExit("threshold/alpha must be in [0,1] and min-area cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false; use --device cpu")
    device = torch.device(args.device)
    model = build_model(args.model, args.encoder_weights, device)
    load_checkpoint_strict(model, args.checkpoint, device)
    model.eval()
    manifest = read_manifest(args.manifest)

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open input video: {args.input_video}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = args.output_fps or (source_fps if source_fps > 0 else 10.0)
    output_width = width * 3 if args.side_by_side else width
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*args.codec), fps, (output_width, height))
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not open output writer: {args.output_video}; try --codec mp4v")

    rows, frame_index, started, stopped = [], 0, time.perf_counter(), False
    try:
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                tensor = preprocess_frame(frame, args.input_size).to(device, non_blocking=True)
                probability = resize_probability(logits_to_probability(model(tensor)), height, width)
                raw_mask = (probability >= args.threshold).astype(np.uint8)
                mask, components = filter_components(raw_mask, args.min_area)
                annotated = annotate_prediction(frame, probability, mask, components, args.mask_alpha,
                                                args.contour_thickness, args.box_thickness)
                entry = manifest.get(frame_index)
                gt = read_gt(args, entry, frame.shape)
                if gt is not None:
                    metrics = binary_metrics(mask, gt)
                    rows.append({"frame_index": frame_index,
                                 "source_filename": entry.get("source_image_filename", "") if entry else "", **metrics})
                    gt_view = frame.copy()
                    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if args.show_ground_truth or args.side_by_side:
                        cv2.drawContours(gt_view if args.side_by_side else annotated, contours, -1, (255, 0, 255), 2)
                    if args.side_by_side:
                        annotated = np.hstack((frame, gt_view, annotated))
                elif args.side_by_side:
                    annotated = np.hstack((frame, frame, annotated))
                writer.write(annotated)
                frame_index += 1
                if frame_index == 1 or frame_index % args.progress_every == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(f"Processed {frame_index} frames ({frame_index / elapsed:.2f} FPS)")
                if args.display:
                    cv2.imshow("Video segmentation", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stopped = True
                        break
    finally:
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if rows:
        csv_path = args.metrics_csv or args.output_video.with_name(args.output_video.stem + "_metrics.csv")
        json_path = args.metrics_json or args.output_video.with_name(args.output_video.stem + "_metrics.json")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=("frame_index", "source_filename", *METRIC_NAMES))
            writer_csv.writeheader(); writer_csv.writerows(rows)
        summary = {name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES}
        summary.update({"evaluated_frames": len(rows), "processed_frames": frame_index})
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Mean metrics: " + ", ".join(f"{name}={summary[name]:.4f}" for name in METRIC_NAMES))
        print(f"Metrics: {csv_path}, {json_path}")
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(f"Finished{' (stopped by user)' if stopped else ''}: {frame_index} frames in {elapsed:.2f}s "
          f"({frame_index / elapsed:.2f} FPS), output: {args.output_video}")


if __name__ == "__main__":
    main()
