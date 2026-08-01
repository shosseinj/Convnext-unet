#!/usr/bin/env python3
"""Create a deterministic MP4 and frame manifest from Kvasir-SEG images."""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def fit_frame(image, width, height, letterbox):
    if not letterbox:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.zeros((height, width, 3), dtype=image.dtype)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="Default: <output stem>_manifest.json")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--letterbox", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the full selected sequence")
    parser.add_argument("--frames-per-image", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.images_dir.is_dir():
        raise SystemExit(f"Images directory does not exist: {args.images_dir}")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0 or args.repeat <= 0 or args.frames_per_image <= 0:
        raise SystemExit("FPS, dimensions, repeat, and frames-per-image must be positive")
    images = sorted((p for p in args.images_dir.iterdir() if p.suffix.lower() in EXTENSIONS), key=lambda p: p.name.lower())
    if args.shuffle:
        random.Random(args.seed).shuffle(images)
    if args.max_images is not None:
        images = images[:args.max_images]
    if not images:
        raise SystemExit(f"No supported images found in: {args.images_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*args.codec), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {args.output}; try --codec mp4v")
    manifest = []
    frame_index = 0
    try:
        for _ in range(args.repeat):
            for image_path in images:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Warning: skipping unreadable image {image_path}", file=sys.stderr)
                    continue
                frame = fit_frame(image, args.width, args.height, args.letterbox)
                gt_name = None
                if args.ground_truth_dir:
                    matches = [p for p in args.ground_truth_dir.glob(image_path.stem + ".*") if p.suffix.lower() in EXTENSIONS]
                    gt_name = matches[0].name if matches else None
                for _ in range(args.frames_per_image):
                    writer.write(frame)
                    manifest.append({"frame_index": frame_index, "source_image_filename": image_path.name,
                                     "ground_truth_mask_filename": gt_name,
                                     "letterbox": args.letterbox})
                    frame_index += 1
    finally:
        writer.release()
    manifest_path = args.manifest or args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"video": str(args.output), "frames": manifest}, indent=2), encoding="utf-8")
    print(f"Created {args.output} with {frame_index} frames; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
