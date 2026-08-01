#!/usr/bin/env python3
"""Generate immutable development split manifests for all protocol seeds."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_pipeline.reproducibility import make_split_manifest, save_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs" / "splits")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407, 2026])
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    for seed in args.seeds:
        manifest = make_split_manifest(args.data_root, ["Kvasir-SEG", "CVC-ClinicDB"], seed,
                                       args.validation_fraction)
        path = args.output_dir / f"development_seed_{seed}.json"
        save_manifest(manifest, path)
        counts = {name: value["counts"] for name, value in manifest["datasets"].items()}
        print(f"Saved {path} sha256={manifest['sha256']} counts={counts}")


if __name__ == "__main__":
    main()
