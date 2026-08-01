#!/usr/bin/env python3
"""Generate control-result artifacts with explicit trained/reused provenance."""

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LARIBPOLYPDB")
FAMILIES = (
    ("MSC branches", ("control_msc_branches_2", "control_msc_branches_3", "control_msc_branches_4", "control_msc_branches_5")),
    ("MSC dilation", ("control_msc_dilations_1_2_3", "control_msc_dilations_1_3_5", "control_msc_dilations_1_3_7")),
    ("Detail width", ("control_detail_0", "control_detail_16", "control_detail_32", "control_detail_64")),
    ("Deep supervision", ("control_ds_0", "control_ds_1", "control_ds_2", "control_ds_3")),
    ("Skip fusion", ("control_skip_normal", "control_skip_attention_gate", "control_skip_bsei")),
    ("Detail fusion", ("control_gdf_addition", "control_gdf_concatenation", "control_gdf_attention", "control_gdf_proposed")),
    ("Backbone", ("control_backbone_resnet34", "control_backbone_efficientnet_b0", "control_backbone_convnext_tiny")),
)


def load_one(results_root, source_variant):
    path = Path(results_root) / "aggregated" / "official" / f"{source_variant}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validated aggregate: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("status") != "PASS" or value.get("variant") != source_variant
            or value.get("seeds") != [42, 3407, 2026]
            or tuple(value.get("datasets", {})) != DATASETS
            or len(value.get("source_summaries", ())) != 15
            or any(len(row.get("sha256", "")) != 64 for row in value["source_summaries"])):
        raise ValueError(f"Aggregate evidence contract failed: {path}")
    return value


def load_controls(results_root, matrix_path):
    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    independent = [row["id"] for row in matrix["controls"]]
    reuse = matrix["control_reuse"]
    expected = {variant for _, variants in FAMILIES for variant in variants}
    if expected != set(independent) | set(reuse):
        raise ValueError("Control family coverage does not match canonical independent/reuse matrix")
    loaded = {}
    for control in independent:
        loaded[control] = {"source_variant": control, "provenance": "independent_trained",
                           "aggregate": load_one(results_root, control)}
    for control, source in reuse.items():
        loaded[control] = {"source_variant": source, "provenance": f"reused:{source}",
                           "aggregate": load_one(results_root, source)}
    return loaded


def generate(loaded, manuscript_root):
    root = Path(manuscript_root)
    for name in ("sections", "tables", "figures", "captions"):
        (root / name).mkdir(parents=True, exist_ok=True)
    rows = []
    for family, variants in FAMILIES:
        for control in variants:
            item = loaded[control]
            for dataset in DATASETS:
                metrics = item["aggregate"]["datasets"][dataset]["metrics"]
                row = {"family": family, "control": control, "source_variant": item["source_variant"],
                       "provenance": item["provenance"], "dataset": dataset}
                for metric in ("dice", "iou"):
                    stats = metrics[metric]
                    for key in ("mean", "sample_std", "ci95_lower", "ci95_upper"):
                        value = stats[key]
                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                            raise ValueError(f"Non-finite value: {control}/{dataset}/{metric}/{key}")
                        row[f"{metric}_{key}"] = value
                    for seed, value in stats["values"].items():
                        row[f"{metric}_seed_{seed}"] = value
                rows.append(row)
    csv_path = root / "tables" / "control_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    lines = ["\\begin{tabular}{lllcc}", "\\hline",
             "Family & Control & Dataset & Dice (mean $\\pm$ SD) & IoU (mean $\\pm$ SD) \\\\", "\\hline"]
    for row in rows:
        control = row["control"].replace("_", "\\_")
        lines.append(f"{row['family']} & {control} & {row['dataset']} & "
                     f"{row['dice_mean']:.4f} $\\pm$ {row['dice_sample_std']:.4f} & "
                     f"{row['iou_mean']:.4f} $\\pm$ {row['iou_sample_std']:.4f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    (root / "tables" / "control_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    colors = ("#3366cc", "#dc3912", "#ff9900", "#109618", "#990099")
    elements = []
    for family_index, (family, variants) in enumerate(FAMILIES):
        top = 45 + family_index * 180
        elements.append(f'<text x="20" y="{top}" font-size="17">{family}</text>')
        for dataset_index, dataset in enumerate(DATASETS):
            points = []
            for index, control in enumerate(variants):
                value = loaded[control]["aggregate"]["datasets"][dataset]["metrics"]["dice"]["mean"]
                x = 90 + index * (620 / max(1, len(variants) - 1))
                y = top + 125 - value * 100
                points.append(f"{x:.1f},{y:.1f}")
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colors[dataset_index]}"/>')
            elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[dataset_index]}" stroke-width="2"/>')
        for index, control in enumerate(variants):
            x = 90 + index * (620 / max(1, len(variants) - 1))
            elements.append(f'<text x="{x:.1f}" y="{top + 150}" text-anchor="middle" font-size="10">{control}</text>')
    for index, dataset in enumerate(DATASETS):
        elements.append(f'<text x="760" y="{55 + index * 24}" fill="{colors[index]}">{dataset}</text>')
    height = 45 + len(FAMILIES) * 180
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="{height}" viewBox="0 0 1050 {height}">'
           f'<rect width="1050" height="{height}" fill="white"/><g font-family="Arial" font-size="13" fill="#111">'
           '<text x="525" y="22" text-anchor="middle" font-size="18">Official three-seed control Dice</text>'
           + "".join(elements) + '</g></svg>')
    (root / "figures" / "control_results.svg").write_text(svg, encoding="utf-8")
    (root / "sections" / "control_results.md").write_text(
        "# Official control results\n\nThe mechanism-isolation table reports per-seed Dice and IoU, the sample "
        "standard deviation, and the prespecified 95% Student-t interval for every independent control. "
        "Computation-identical configurations are explicitly linked to their cumulative-run source and are not "
        "presented as separately trained evidence. Numerical interpretations must remain descriptive unless a "
        "separately validated significance analysis is reported.\n", encoding="utf-8")
    (root / "captions" / "control_results.md").write_text(
        "**Table X. Official mechanism-isolation controls.** Independent configurations are trained with seeds "
        "42, 3407, and 2026. Rows marked as reused inherit the hash-bound aggregate of a computation-identical "
        "cumulative configuration. **Figure X.** Three-seed mean Dice across each prespecified control family and "
        "evaluation dataset; monotonic behavior is not assumed.\n", encoding="utf-8")
    return {"rows": len(rows), "families": len(FAMILIES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs" / "ablation_matrix.yaml")
    parser.add_argument("--manuscript-root", type=Path, default=ROOT / "manuscript")
    args = parser.parse_args()
    try:
        loaded = load_controls(args.results_root, args.matrix)
        counts = generate(loaded, args.manuscript_root)
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise SystemExit(f"Control manuscript generation BLOCKED: {error}") from None
    print(json.dumps({"status": "PASS", **counts}))


if __name__ == "__main__":
    main()
