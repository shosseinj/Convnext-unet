#!/usr/bin/env python3
"""Aggregate seed-42 Pilot summaries without treating them as official results."""

import csv
import json
from pathlib import Path


VARIANTS = ("baseline", "baseline_msc", "baseline_msc_bsei", "baseline_msc_bsei_detail",
            "baseline_msc_bsei_detail_gdf", "full")


def main():
    root = Path(__file__).resolve().parents[1]
    rows = []
    for variant in VARIANTS:
        run = root / "results" / "raw" / variant / "seed_42" / "pilot"
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        best = max(history, key=lambda item: item["selection_dice"])
        rows.append({"variant": variant, "status": summary["status"], "epochs": summary["epochs_completed"],
                     "best_epoch": best["epoch"], "selection_dice": best["selection_dice"],
                     "kvasir_dice": best["validation"]["Kvasir-SEG"]["dice"],
                     "clinicdb_dice": best["validation"]["CVC-ClinicDB"]["dice"],
                     "kvasir_soft_dice": best["validation"]["Kvasir-SEG"]["soft_dice"],
                     "clinicdb_soft_dice": best["validation"]["CVC-ClinicDB"]["soft_dice"],
                     "elapsed_seconds": summary["elapsed_seconds"], "eligible_for_manuscript": False})
    output_dir = root / "results" / "aggregated"; output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_seed42.json").write_text(json.dumps({"status": "NEEDS_REVIEW", "rows": rows}, indent=2), encoding="utf-8")
    with (output_dir / "pilot_seed42.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['variant']}: best_epoch={row['best_epoch']} selection_dice={row['selection_dice']:.4f}")


if __name__ == "__main__":
    main()
