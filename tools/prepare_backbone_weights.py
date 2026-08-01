#!/usr/bin/env python3
"""Download and verify official torchvision ImageNet weights for backbone controls."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchvision.models import (EfficientNet_B0_Weights, ResNet34_Weights,
                                efficientnet_b0, resnet34)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    output = ROOT / "pretrained"
    output.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("resnet34", ResNet34_Weights.DEFAULT.url, resnet34),
        ("efficientnet_b0", EfficientNet_B0_Weights.DEFAULT.url, efficientnet_b0),
    )
    records = []
    for name, url, constructor in specifications:
        filename = url.rsplit("/", 1)[-1]
        path = output / filename
        state = torch.hub.load_state_dict_from_url(
            url, model_dir=str(output), map_location="cpu", check_hash=True, progress=True
        )
        model = constructor(weights=None)
        model.load_state_dict(state, strict=True)
        records.append({"backbone": name, "url": url, "path": str(path),
                        "bytes": path.stat().st_size, "sha256": sha256_file(path),
                        "strict_load": True})
    manifest = {"source": "official torchvision weight enums", "records": records,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
