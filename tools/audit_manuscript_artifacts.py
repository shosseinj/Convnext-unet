#!/usr/bin/env python3
"""Audit prepared manuscript artifacts without promoting incomplete results."""

import csv
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "audit": ("section", "table_csv", "table_tex", "figure", "caption"),
    "bsei": ("section", "table_csv", "table_tex", "figure", "caption"),
    "loss": ("section", "table_csv", "table_tex", "figure", "caption"),
    "experiment": ("section", "table_csv", "table_tex", "figure", "caption"),
    "complexity": ("section", "table_csv", "table_tex", "figure", "caption"),
    "controls": ("section", "table_csv", "table_tex", "figure", "caption"),
    "limitations": ("section", "table_csv", "table_tex", "figure", "caption"),
}
PATHS = {
    "section": "sections/{stage}.md",
    "table_csv": "tables/{stage}.csv",
    "table_tex": "tables/{stage}.tex",
    "figure": "figures/{stage}.svg",
    "caption": "captions/{stage}.md",
}
OVERRIDES = {
    ("limitations", "table_csv"): "tables/reporting_decisions.csv",
    ("limitations", "table_tex"): "tables/reporting_decisions.tex",
    ("limitations", "figure"): "figures/evidence_boundaries.svg",
}
FORBIDDEN = re.compile(r"\b(?:TBD|XX|LRSE)\b|LRS-E|\{\{[^}]+\}\}", re.IGNORECASE)


def audit(root):
    root = Path(root)
    stage_reports = []
    for stage, kinds in STAGES.items():
        files = []
        for kind in kinds:
            relative = OVERRIDES.get((stage, kind), PATHS[kind].format(stage=stage))
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty artifact: {path}")
            if path.suffix == ".svg":
                ET.parse(path)
            if path.suffix == ".csv":
                with path.open(encoding="utf-8") as handle:
                    if not list(csv.DictReader(handle)):
                        raise ValueError(f"CSV has no data rows: {path}")
            if path.suffix in {".md", ".tex", ".csv"}:
                match = FORBIDDEN.search(path.read_text(encoding="utf-8"))
                if match:
                    raise ValueError(f"Forbidden placeholder/legacy term '{match.group(0)}' in {path}")
            files.append(str(path))
        stage_reports.append({"stage": stage, "status": "PASS", "files": files})
    numerical_packages = {
        "incremental_results": ("sections/incremental_results.md", "tables/incremental_results.csv",
                                "tables/incremental_results.tex", "figures/incremental_results.svg",
                                "captions/incremental_results.md"),
        "control_results": ("sections/control_results.md", "tables/control_results.csv",
                            "tables/control_results.tex", "figures/control_results.svg",
                            "captions/control_results.md"),
    }
    numerical_reports = []
    for name, relatives in numerical_packages.items():
        trigger = root / relatives[1]
        if not trigger.is_file():
            numerical_reports.append({"package": name, "status": "BLOCKED_BY_EXPERIMENT"})
            continue
        files = []
        for relative in relatives:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Incomplete numerical artifact package: {path}")
            if path.suffix == ".svg":
                ET.parse(path)
            if path.suffix == ".csv":
                with path.open(encoding="utf-8") as handle:
                    if not list(csv.DictReader(handle)):
                        raise ValueError(f"Numerical CSV has no rows: {path}")
            files.append(str(path))
        numerical_reports.append({"package": name, "status": "PASS", "files": files})

    selection_path = root / "figures" / "qualitative_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "PASS" or selection.get("selection_uses_prediction_performance") is not False:
            raise ValueError("Qualitative selection evidence contract failed")
        figures = selection.get("figures", ())
        if len(figures) != 5:
            raise ValueError("Qualitative package must contain five dataset figures")
        for item in figures:
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing qualitative figure: {path}")
            with Image.open(path) as image:
                image.verify()
        caption = root / "captions" / "qualitative.md"
        if not caption.is_file() or caption.stat().st_size == 0:
            raise ValueError("Missing qualitative caption")
        qualitative_status = "PASS"
    else:
        qualitative_status = "BLOCKED_BY_EVALUATION"
    all_numerical_pass = (all(row["status"] == "PASS" for row in numerical_reports)
                          and qualitative_status == "PASS")
    return {"status": "PASS", "scope": "non_numerical_manuscript_artifacts",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stages": stage_reports, "numerical_packages": numerical_reports,
            "qualitative_package": qualitative_status,
            "numerical_results": "READY" if all_numerical_pass else "BLOCKED_BY_EXPERIMENT"}


def main():
    report = audit(ROOT / "manuscript")
    output = ROOT / "reports" / "manuscript_artifact_gate.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "stages": len(report["stages"]),
                      "numerical_results": report["numerical_results"]}))


if __name__ == "__main__":
    main()
