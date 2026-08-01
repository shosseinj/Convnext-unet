"""Create a verified source-DOCX backup only after the Word gate is READY."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

try:
    from .check_word_update_readiness import assess_readiness, sha256
except ImportError:  # direct-script execution
    from check_word_update_readiness import assess_readiness, sha256


def create_backup(root: Path, backup_dir: Path, timestamp: str | None = None) -> dict:
    root = root.resolve()
    readiness = assess_readiness(root)
    if readiness["status"] != "READY":
        raise RuntimeError(
            "Word update readiness is BLOCKED: "
            + ", ".join(item["id"] for item in readiness["blockers"])
        )

    source = Path(readiness["source"]["path"])
    source_hash = readiness["source"]["sha256"]
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_dir if backup_dir.is_absolute() else root / backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / f"{source.stem}.before_word_update.{stamp}{source.suffix}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {destination}")

    shutil.copy2(source, destination)
    destination_hash = sha256(destination)
    if destination_hash != source_hash:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Backup hash mismatch; incomplete backup removed")

    manifest = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": source_hash,
        "backup": str(destination),
        "backup_sha256": destination_hash,
        "bytes": destination.stat().st_size,
        "readiness_status": readiness["status"],
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backup-dir", type=Path, default=Path("manuscript/backups"))
    args = parser.parse_args()
    try:
        manifest = create_backup(args.root, args.backup_dir)
    except RuntimeError as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}))
        return 2
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
