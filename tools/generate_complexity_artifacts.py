#!/usr/bin/env python3
"""Generate manuscript complexity artifacts only from the passed architecture gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "baseline": "Baseline",
    "baseline_msc": "+MSC",
    "baseline_msc_bsei": "+BSEI",
    "baseline_msc_bsei_detail": "+Detail Branch",
    "baseline_msc_bsei_detail_gdf": "+GDF",
    "full": "+Deep Supervision",
}


def main():
    source = ROOT / "reports" / "architecture_gate.json"
    gate = json.loads(source.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("input_size") != [352, 352]:
        raise SystemExit("Architecture gate is not eligible for manuscript artifacts")
    variants = gate.get("variants", [])
    if [row.get("variant") for row in variants] != list(LABELS):
        raise SystemExit("Architecture gate variant order/coverage mismatch")
    if any(row.get("status") != "PASS" for row in variants):
        raise SystemExit("At least one architecture variant did not pass")

    table_dir = ROOT / "manuscript" / "tables"
    figure_dir = ROOT / "manuscript" / "figures"
    caption_dir = ROOT / "manuscript" / "captions"
    section_dir = ROOT / "manuscript" / "sections"
    for directory in (table_dir, figure_dir, caption_dir, section_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_rows = [{"variant": LABELS[row["variant"]], "parameters": row["parameters"],
                 "macs": row["macs"], "gflops_2x_macs": row["gflops_convention_2x_macs"]}
                for row in variants]
    with (table_dir / "complexity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader(); writer.writerows(csv_rows)
    lines = ["\\begin{tabular}{lrrr}", "\\hline",
             "Variant & Parameters & MACs & GFLOPs ($2\\times$MACs) \\\\", "\\hline"]
    for row in csv_rows:
        lines.append(f"{row['variant']} & {row['parameters']:,} & {row['macs']:,} & "
                     f"{row['gflops_2x_macs']:.3f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    (table_dir / "complexity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    max_flops = max(row["gflops_2x_macs"] for row in csv_rows)
    bars = []
    for index, row in enumerate(csv_rows):
        y = 45 + index * 42
        width = 560 * row["gflops_2x_macs"] / max_flops
        bars.append(
            f'<text x="165" y="{y + 18}" text-anchor="end">{row["variant"]}</text>'
            f'<rect x="180" y="{y}" width="{width:.2f}" height="24" fill="#4776b4"/>'
            f'<text x="{190 + width:.2f}" y="{y + 18}">{row["gflops_2x_macs"]:.3f}</text>'
        )
    svg = ("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"850\" height=\"320\" "
           "viewBox=\"0 0 850 320\"><rect width=\"850\" height=\"320\" fill=\"white\"/>"
           "<g font-family=\"Arial\" font-size=\"14\" fill=\"#111\">"
           "<text x=\"425\" y=\"24\" text-anchor=\"middle\" font-size=\"18\">"
           "Architecture complexity at 352 x 352 (GFLOPs = 2 x MACs)</text>"
           + "".join(bars) + "</g></svg>")
    (figure_dir / "complexity.svg").write_text(svg, encoding="utf-8")
    (caption_dir / "complexity.md").write_text(
        "Computational complexity of the six incremental BSEI-ConvNeXt-UNet variants at an "
        "input size of 352 by 352 pixels. FLOPs are reported using the explicit convention "
        "FLOPs = 2 x MACs. Values are architecture measurements and do not imply segmentation "
        "performance. Evidence: reports/architecture_gate.json.\n", encoding="utf-8")
    (section_dir / "complexity.md").write_text(
        "# Complexity text ready for Word review\n\n"
        "The incremental variants contain approximately 29.19-29.60 million parameters under "
        "the audited implementation. At 352 by 352 input resolution, the corresponding "
        "complexity is 27.202-28.905 GFLOPs when one multiply-accumulate is counted as two "
        "floating-point operations. These values describe architecture cost only; performance "
        "comparisons remain pending until all official runs are validated.\n", encoding="utf-8")
    print("Generated complexity CSV, LaTeX, SVG, caption, and section text")


if __name__ == "__main__":
    main()
