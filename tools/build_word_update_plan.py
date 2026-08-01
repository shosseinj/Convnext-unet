"""Materialize a hash-bound paragraph update plan after the Word gate is READY."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from .check_word_update_readiness import assess_readiness, sha256
except ImportError:  # direct-script execution
    from check_word_update_readiness import assess_readiness, sha256


def _markdown_body(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.match(r"^#{1,6}\s+", lines[0]):
        lines.pop(0)
    body = "\n".join(lines).strip()
    return [re.sub(r"\s*\n\s*", " ", block).strip()
            for block in re.split(r"\n\s*\n", body) if block.strip()]


def materialize_target(root: Path, target: dict) -> dict:
    action = target["action"]
    item = {"paragraph": target["paragraph"], "action": action}
    if action == "delete":
        item["replacement_paragraphs"] = []
        return item
    if "replacement_text" in target:
        item["replacement_paragraphs"] = [target["replacement_text"].strip()]
        item["payload_kind"] = "inline_text"
        return item

    relative = target["replacement_source"]
    source = root / relative
    mode = target["replacement_mode"]
    item.update({"payload_kind": mode, "replacement_source": relative,
                 "replacement_source_sha256": sha256(source)})
    if mode == "markdown_body":
        item["replacement_paragraphs"] = _markdown_body(source.read_text(encoding="utf-8"))
    else:
        item["replacement_paragraphs"] = []
        item["source_reference"] = relative
    return item


def build_plan(root: Path) -> dict:
    root = root.resolve()
    readiness = assess_readiness(root)
    if readiness["status"] != "READY":
        raise RuntimeError("Word update readiness is BLOCKED: " + ", ".join(
            blocker["id"] for blocker in readiness["blockers"]))
    target_path = root / "reports" / "docx_target_paragraphs.json"
    target_map = json.loads(target_path.read_text(encoding="utf-8"))
    actions = [materialize_target(root, target) for target in target_map["targets"]]
    if any(action["action"] == "replace" and action.get("payload_kind") != "source_reference"
           and not action["replacement_paragraphs"] for action in actions):
        raise RuntimeError("A replacement materialized to an empty paragraph list")
    return {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_docx": readiness["source"],
        "target_map": str(target_path),
        "target_map_sha256": sha256(target_path),
        "action_count": len(actions),
        "actions": actions,
        "global_replacements": target_map.get("global_replacements", []),
        "structured_updates": target_map.get("structured_updates", []),
        "next_action": "Create the verified backup, apply this plan to a copy, then render and inspect every page.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("reports/word_update_plan.json"))
    args = parser.parse_args()
    try:
        plan = build_plan(args.root)
    except RuntimeError as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}))
        return 2
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "actions": plan["action_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
