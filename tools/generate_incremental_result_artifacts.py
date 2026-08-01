#!/usr/bin/env python3
"""Generate manuscript artifacts exclusively from validated official aggregates."""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("baseline", "baseline_msc", "baseline_msc_bsei", "baseline_msc_bsei_detail",
            "baseline_msc_bsei_detail_gdf", "full")
DATASETS = ("Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LARIBPOLYPDB")
LABELS = {"baseline": "Baseline", "baseline_msc": "+MSC", "baseline_msc_bsei": "+BSEI",
          "baseline_msc_bsei_detail": "+Detail", "baseline_msc_bsei_detail_gdf": "+GDF",
          "full": "+Deep supervision"}


def load_aggregates(results_root):
    loaded = {}
    for variant in VARIANTS:
        path = Path(results_root) / "aggregated" / "official" / f"{variant}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing validated aggregate: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("status") != "PASS" or value.get("variant") != variant
                or value.get("seeds") != [42, 3407, 2026]
                or tuple(value.get("datasets", {})) != DATASETS
                or len(value.get("source_summaries", ())) != 15
                or any(len(row.get("sha256", "")) != 64 for row in value["source_summaries"])):
            raise ValueError(f"Aggregate evidence contract failed: {path}")
        for dataset in DATASETS:
            for metric in ("dice", "iou"):
                stats = value["datasets"][dataset]["metrics"][metric]
                numbers = [stats.get(key) for key in ("mean", "sample_std", "ci95_lower", "ci95_upper")]
                if any(not isinstance(number, (int, float)) or not math.isfinite(number) for number in numbers):
                    raise ValueError(f"Non-finite aggregate statistic: {variant}/{dataset}/{metric}")
        loaded[variant] = value
    return loaded


def generate(loaded, manuscript_root):
    manuscript_root = Path(manuscript_root)
    for name in ("sections", "tables", "figures", "captions"):
        (manuscript_root / name).mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            metrics = loaded[variant]["datasets"][dataset]["metrics"]
            row = {"variant": variant, "dataset": dataset}
            for metric in ("dice", "iou"):
                for key in ("mean", "sample_std", "ci95_lower", "ci95_upper"):
                    row[f"{metric}_{key}"] = metrics[metric][key]
                for seed, value in metrics[metric]["values"].items():
                    row[f"{metric}_seed_{seed}"] = value
            rows.append(row)
    csv_path = manuscript_root / "tables" / "incremental_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    lines = ["\\begin{tabular}{llcc}", "\\hline",
             "Variant & Dataset & Dice (mean $\\pm$ SD) & IoU (mean $\\pm$ SD) \\\\", "\\hline"]
    for row in rows:
        lines.append(f"{LABELS[row['variant']]} & {row['dataset']} & "
                     f"{row['dice_mean']:.4f} $\\pm$ {row['dice_sample_std']:.4f} & "
                     f"{row['iou_mean']:.4f} $\\pm$ {row['iou_sample_std']:.4f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    (manuscript_root / "tables" / "incremental_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    colors = ("#3366cc", "#dc3912", "#ff9900", "#109618", "#990099")
    elements = []
    for dataset_index, dataset in enumerate(DATASETS):
        points = []
        for index, variant in enumerate(VARIANTS):
            value = loaded[variant]["datasets"][dataset]["metrics"]["dice"]["mean"]
            x = 100 + index * 125
            y = 410 - value * 350
            points.append(f"{x:.1f},{y:.1f}")
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[dataset_index]}"/>')
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[dataset_index]}" stroke-width="2"/>')
        elements.append(f'<text x="790" y="{65 + dataset_index * 24}" fill="{colors[dataset_index]}">{dataset}</text>')
    for index, variant in enumerate(VARIANTS):
        elements.append(f'<text x="{100 + index * 125}" y="438" text-anchor="middle">{LABELS[variant]}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="950" height="470" viewBox="0 0 950 470">'
           '<rect width="950" height="470" fill="white"/><g font-family="Arial" font-size="13" fill="#111">'
           '<text x="475" y="25" text-anchor="middle" font-size="18">Official three-seed Dice by incremental variant</text>'
           '<line x1="75" y1="45" x2="75" y2="410" stroke="#333"/><line x1="75" y1="410" x2="760" y2="410" stroke="#333"/>'
           + "".join(elements) + '</g></svg>')
    (manuscript_root / "figures" / "incremental_results.svg").write_text(svg, encoding="utf-8")

    size_rows = []
    full = loaded["full"]
    for dataset in DATASETS:
        for size_bin, metrics in full["datasets"][dataset]["size_stratified"].items():
            for metric in ("dice", "iou"):
                stats = metrics[metric]
                size_rows.append({"variant": "full", "dataset": dataset, "size_bin": size_bin,
                                  "metric": metric, "mean": stats["mean"],
                                  "sample_std": stats["sample_std"],
                                  "ci95_lower": stats["ci95_lower"], "ci95_upper": stats["ci95_upper"]})
    with (manuscript_root / "tables" / "size_stratified_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=size_rows[0].keys())
        writer.writeheader(); writer.writerows(size_rows)

    best_lines = []
    for dataset in DATASETS:
        best_variant = max(VARIANTS, key=lambda item: loaded[item]["datasets"][dataset]["metrics"]["dice"]["mean"])
        stats = loaded[best_variant]["datasets"][dataset]["metrics"]["dice"]
        best_lines.append(f"For {dataset}, the highest observed three-seed mean Dice was "
                          f"{stats['mean']:.4f} (sample SD {stats['sample_std']:.4f}) for {LABELS[best_variant]}.")
    section = ("# Official incremental results\n\n"
               "All values below were generated from validated official checkpoints using the fixed threshold and "
               "prespecified three seeds. They are descriptive observations and do not by themselves establish "
               "statistical significance or causal effects.\n\n" + "\n\n".join(best_lines) + "\n")
    (manuscript_root / "sections" / "incremental_results.md").write_text(section, encoding="utf-8")
    (manuscript_root / "captions" / "incremental_results.md").write_text(
        "**Table X.** Official Dice and IoU for the six incremental configurations, reported as mean and sample "
        "standard deviation across seeds 42, 3407, and 2026. **Figure X.** Three-seed mean Dice across the "
        "incremental sequence for the five prespecified evaluation datasets. The sequence is not assumed to be "
        "monotonic. Evidence: results/aggregated/official/*.json.\n", encoding="utf-8")
    return {"rows": len(rows), "size_rows": len(size_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--manuscript-root", type=Path, default=ROOT / "manuscript")
    args = parser.parse_args()
    try:
        loaded = load_aggregates(args.results_root)
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise SystemExit(f"Incremental manuscript generation BLOCKED: {error}") from None
    counts = generate(loaded, args.manuscript_root)
    print(json.dumps({"status": "PASS", **counts}))


if __name__ == "__main__":
    main()
