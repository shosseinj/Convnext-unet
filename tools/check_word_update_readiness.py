"""Fail-closed readiness gate for the controlled final DOCX update."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTHOR_FIELDS = (
    "authors",
    "affiliations",
    "corresponding_author",
    "code_availability",
    "credit_roles",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(_nonempty(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(_nonempty(item) for item in value.values())
    return value is not None


def assess_readiness(root: Path, source_override: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    target_path = root / "reports" / "docx_target_paragraphs.json"
    artifact_path = root / "reports" / "manuscript_artifact_gate.json"
    author_path = root / "manuscript" / "author_inputs.json"
    blockers: list[dict[str, Any]] = []

    if not target_path.exists():
        blockers.append({"id": "TARGET_MAP_MISSING", "path": str(target_path)})
        targets = {"targets": [], "source": "", "source_sha256": ""}
    else:
        targets = json.loads(target_path.read_text(encoding="utf-8"))

    source = source_override or Path(targets.get("source", ""))
    source_evidence: dict[str, Any] = {"path": str(source), "exists": source.exists()}
    if not source.exists():
        blockers.append({"id": "SOURCE_DOCX_MISSING", "path": str(source)})
    else:
        actual_hash = sha256(source)
        expected_hash = targets.get("source_sha256")
        source_evidence.update({"sha256": actual_hash, "expected_sha256": expected_hash})
        if actual_hash != expected_hash:
            blockers.append({
                "id": "SOURCE_DOCX_HASH_MISMATCH",
                "expected": expected_hash,
                "actual": actual_hash,
            })

    status_counts: dict[str, int] = {}
    unresolved_targets = []
    invalid_targets = []
    seen_paragraphs: set[int] = set()
    body_paragraph_count = targets.get("body_paragraph_count")
    for target in targets.get("targets", []):
        paragraph = target.get("paragraph")
        action = target.get("action")
        target_errors: list[str] = []
        if not isinstance(paragraph, int) or paragraph < 0:
            target_errors.append("paragraph must be a non-negative integer")
        elif paragraph in seen_paragraphs:
            target_errors.append("paragraph target is duplicated")
        else:
            seen_paragraphs.add(paragraph)
        if (
            isinstance(body_paragraph_count, int)
            and isinstance(paragraph, int)
            and paragraph >= body_paragraph_count
        ):
            target_errors.append("paragraph is outside body_paragraph_count")
        if action not in {"replace", "delete"}:
            target_errors.append("action must be replace or delete")

        status = target.get("status", "MISSING_STATUS")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "READY_FOR_WORD" and action == "replace":
            replacement_text = target.get("replacement_text")
            replacement_source = target.get("replacement_source")
            if bool(replacement_text) == bool(replacement_source):
                target_errors.append(
                    "replace requires exactly one of replacement_text or replacement_source"
                )
            elif replacement_source:
                replacement_mode = target.get("replacement_mode")
                if replacement_mode not in {"markdown_body", "source_reference"}:
                    target_errors.append(
                        "replacement_source requires markdown_body or source_reference mode"
                    )
                source_path = root / replacement_source
                if not source_path.is_file():
                    target_errors.append(f"replacement_source is missing: {replacement_source}")
                elif not source_path.read_text(encoding="utf-8").strip():
                    target_errors.append(f"replacement_source is empty: {replacement_source}")
        if target_errors:
            invalid_targets.append({"paragraph": paragraph, "errors": target_errors})
        if status != "READY_FOR_WORD":
            unresolved_targets.append({
                "paragraph": target.get("paragraph"),
                "status": status,
                "required": target.get("required"),
            })
    if unresolved_targets:
        blockers.append({"id": "UNRESOLVED_DOCX_TARGETS", "targets": unresolved_targets})
    if invalid_targets:
        blockers.append({"id": "INVALID_DOCX_TARGETS", "targets": invalid_targets})

    invalid_structured = []
    for index, update in enumerate(targets.get("structured_updates", [])):
        errors = []
        if update.get("kind") != "table_cell_patch":
            errors.append("unsupported structured update kind")
        if not isinstance(update.get("table_index"), int) or update.get("table_index") < 0:
            errors.append("table_index must be a non-negative integer")
        if not isinstance(update.get("anchor_after_paragraph"), int):
            errors.append("anchor_after_paragraph must be an integer")
        cells = update.get("cells")
        if not isinstance(cells, list) or not cells:
            errors.append("table patch requires at least one cell")
        else:
            for cell in cells:
                if (not isinstance(cell.get("row"), int) or cell["row"] < 0
                        or not isinstance(cell.get("column"), int) or cell["column"] < 0
                        or not _nonempty(cell.get("expected"))
                        or not _nonempty(cell.get("text"))):
                    errors.append("every table cell needs row/column and non-empty expected/text values")
                    break
        evidence = update.get("evidence")
        if evidence and not (root / evidence).is_file():
            errors.append(f"structured-update evidence is missing: {evidence}")
        if errors:
            invalid_structured.append({"index": index, "errors": errors})
    if invalid_structured:
        blockers.append({"id": "INVALID_STRUCTURED_UPDATES", "updates": invalid_structured})

    invalid_globals = []
    for index, replacement in enumerate(targets.get("global_replacements", [])):
        if (not _nonempty(replacement.get("find"))
                or not _nonempty(replacement.get("replace"))
                or replacement.get("find") == replacement.get("replace")):
            invalid_globals.append(index)
    if invalid_globals:
        blockers.append({"id": "INVALID_GLOBAL_REPLACEMENTS", "indices": invalid_globals})

    if not artifact_path.exists():
        artifact = {}
        blockers.append({"id": "MANUSCRIPT_ARTIFACT_GATE_MISSING", "path": str(artifact_path)})
    else:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("status") != "PASS":
            blockers.append({"id": "MANUSCRIPT_ARTIFACT_GATE_NOT_PASS", "status": artifact.get("status")})
        if artifact.get("numerical_results") != "READY":
            blockers.append({"id": "NUMERICAL_RESULTS_NOT_READY", "status": artifact.get("numerical_results")})
        if artifact.get("qualitative_package") != "PASS":
            blockers.append({"id": "QUALITATIVE_PACKAGE_NOT_READY", "status": artifact.get("qualitative_package")})

    if not author_path.exists():
        author = {}
        blockers.append({"id": "AUTHOR_INPUTS_MISSING", "path": str(author_path)})
    else:
        author = json.loads(author_path.read_text(encoding="utf-8"))
        missing_fields = [field for field in AUTHOR_FIELDS if not _nonempty(author.get(field))]
        if missing_fields:
            blockers.append({"id": "AUTHOR_INPUTS_INCOMPLETE", "fields": missing_fields})

    return {
        "status": "READY" if not blockers else "BLOCKED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_evidence,
        "target_status_counts": status_counts,
        "manuscript_gate": {
            "status": artifact.get("status"),
            "numerical_results": artifact.get("numerical_results"),
            "qualitative_package": artifact.get("qualitative_package"),
        },
        "author_inputs_path": str(author_path),
        "required_author_fields": list(AUTHOR_FIELDS),
        "blockers": blockers,
        "next_action": (
            "Create a timestamped backup, apply only mapped replacements, then render and inspect every page."
            if not blockers
            else "Resolve every blocker; do not modify the source DOCX."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/word_update_readiness.json"))
    args = parser.parse_args()
    report = assess_readiness(args.root, args.source)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "blockers": [b["id"] for b in report["blockers"]]}))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
