#!/usr/bin/env python3
"""Generate control-complexity artifacts from the passed 352px control gate."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "reports" / "control_architecture_gate.json"
    gate = json.loads(source.read_text(encoding="utf-8"))
    variants = gate.get("variants", [])
    if (gate.get("status") != "PASS" or gate.get("input_size") != [352, 352]
            or gate.get("matrix_sections") != ["controls"] or len(variants) != 14
            or any(row.get("status") != "PASS" for row in variants)):
        raise SystemExit("Control architecture gate is not manuscript eligible")
    rows = [{"control": row["variant"], "parameters": row["parameters"],
             "macs": row["macs"], "gflops_2x_macs": row["gflops_convention_2x_macs"]}
            for row in variants]
    table_dir = ROOT / "manuscript" / "tables"
    figure_dir = ROOT / "manuscript" / "figures"
    caption_dir = ROOT / "manuscript" / "captions"
    with (table_dir / "control_complexity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    lines = ["\\begin{tabular}{lrrr}", "\\hline",
             "Control & Parameters & MACs & GFLOPs ($2\\times$MACs) \\\\", "\\hline"]
    for row in rows:
        latex_label = row["control"].replace("_", "\\_")
        lines.append(f"{latex_label} & {row['parameters']:,} & "
                     f"{row['macs']:,} & {row['gflops_2x_macs']:.3f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    (table_dir / "control_complexity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    max_value = max(row["gflops_2x_macs"] for row in rows)
    bars = []
    for index, row in enumerate(rows):
        y = 48 + index * 30
        width = 480 * row["gflops_2x_macs"] / max_value
        bars.append(f'<text x="285" y="{y + 16}" text-anchor="end">{row["control"]}</text>'
                    f'<rect x="300" y="{y}" width="{width:.2f}" height="20" fill="#6b5ca5"/>'
                    f'<text x="{310 + width:.2f}" y="{y + 16}">{row["gflops_2x_macs"]:.3f}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="950" height="500" viewBox="0 0 950 500">'
           '<rect width="950" height="500" fill="white"/><g font-family="Arial" font-size="13" fill="#111">'
           '<text x="475" y="25" text-anchor="middle" font-size="18">Control complexity at 352 x 352 (GFLOPs = 2 x MACs)</text>'
           + "".join(bars) + '</g></svg>')
    (figure_dir / "control_complexity.svg").write_text(svg, encoding="utf-8")
    (caption_dir / "control_complexity.md").write_text(
        "Architecture complexity of the 14 independent special controls at 352 by 352 pixels. "
        "FLOPs are defined as twice the measured MAC count. These measurements describe "
        "computational cost only and contain no segmentation-performance claim. Evidence: "
        "reports/control_architecture_gate.json.\n", encoding="utf-8")
    print("Generated control-complexity CSV, LaTeX, SVG and caption")


if __name__ == "__main__":
    main()
