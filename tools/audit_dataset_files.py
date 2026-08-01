#!/usr/bin/env python3
"""Audit image/mask readability, modes, sizes, and TIFF metadata."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2


def audit_dataset(root, name):
    result = {"dataset": name, "pairs": 0, "unreadable": [], "size_mismatches": [],
              "image_modes": Counter(), "mask_modes": Counter(), "image_extensions": Counter(),
              "mask_extensions": Counter(), "detected_formats": Counter(),
              "images_with_alpha_or_extra_channels": [], "extension_format_mismatches": []}
    image_dir, mask_dir = root / name / "images", root / name / "masks"
    masks = {path.name.lower(): path for path in mask_dir.iterdir() if path.is_file()}
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        mask_path = masks.get(image_path.name.lower())
        if mask_path is None:
            result["unreadable"].append({"image": image_path.name, "error": "matching mask missing"})
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            result["unreadable"].append({"image": image_path.name, "error": "OpenCV decode failed"})
            continue
        header = image_path.read_bytes()[:4]
        detected_format = "TIFF" if header[:4] in {b"II*\x00", b"MM\x00*"} else ("JPEG" if header[:2] == b"\xff\xd8" else ("PNG" if header == b"\x89PNG" else "UNKNOWN"))
        image_mode = "L" if image.ndim == 2 else f"{image.shape[2]}-channel"
        mask_mode = "L" if mask.ndim == 2 else f"{mask.shape[2]}-channel"
        image_size, mask_size = (image.shape[1], image.shape[0]), (mask.shape[1], mask.shape[0])
        result["pairs"] += 1
        result["image_modes"][image_mode] += 1; result["mask_modes"][mask_mode] += 1
        result["image_extensions"][image_path.suffix.lower()] += 1
        result["mask_extensions"][mask_path.suffix.lower()] += 1
        result["detected_formats"][detected_format] += 1
        if detected_format == "TIFF" and image_path.suffix.lower() not in {".tif", ".tiff"}:
            result["extension_format_mismatches"].append({"file": image_path.name, "extension": image_path.suffix.lower(), "detected": detected_format})
        if image.ndim == 3 and image.shape[2] not in {1, 3}:
            result["images_with_alpha_or_extra_channels"].append({"file": image_path.name, "mode": image_mode})
        if image_size != mask_size:
            result["size_mismatches"].append({"file": image_path.name, "image_size": image_size, "mask_size": mask_size})
    for key in ("image_modes", "mask_modes", "image_extensions", "mask_extensions", "detected_formats"):
        result[key] = dict(result[key])
    result["status"] = "PASS" if not result["unreadable"] and not result["size_mismatches"] else "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=["Kvasir-SEG", "CVC-ClinicDB"])
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_file_audit.json"))
    args = parser.parse_args()
    results = [audit_dataset(args.data_root, name) for name in args.datasets]
    report = {"status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
              "datasets": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in results:
        print(f"{item['dataset']}: {item['status']} pairs={item['pairs']} "
              f"formats={item['detected_formats']} extension_mismatches={len(item['extension_format_mismatches'])} "
              f"unreadable={len(item['unreadable'])} size_mismatches={len(item['size_mismatches'])}")
    print(f"Report: {args.output}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
