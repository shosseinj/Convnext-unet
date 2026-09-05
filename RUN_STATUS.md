Status: QUALITATIVE ABLATION PACKAGE COMPLETE
Datasets: Kvasir-SEG, CVC-ClinicDB, CVC-300, CVC-ColonDB, ETIS-LaribPolypDB
Samples: exactly 2 valid common IDs per dataset
Selection: deterministic GT-area quantiles; prediction scores are not used
Models: seed-42 Exp01 baseline, Exp33 second, Exp45 third
Validation: matching IDs, readable files, and equal native dimensions
Individual outputs: RGB original plus three binary masks per sample
Expected count: 2 x 5 x 4 = 40 individual PNGs
Panels: 10 four-column panels; optional 10 five-column GT panels
Manifest: qualitative_ablation_outputs/manifest.csv
Generator: tools/generate_qualitative_ablation_examples.py
Instructions: docs/QUALITATIVE_ABLATION.md
Generated: 40 individual PNGs, 10 panels, 10 GT panels, 10 manifest rows
Master paper figure: all_datasets_comparison_panel_with_gt.png (4088x5955, 300 DPI)
