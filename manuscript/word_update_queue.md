# Word Update Queue

Source DOCX: `C:\Users\jafari.h\Downloads\Telegram Desktop\hossein_paper_revised.docx`

## WQ-001 — terminology consistency

- Target: title and body paragraphs 5, 17, 31, 37, 47, 49, 63, 64, 66, 108,
  202, 207, 216, 217, 218, 222, 227, 252, 253, 255, 258, 259 and 261;
  includes Abstract, contributions, Related Work, architecture, decoder,
  ablation controls, figures/captions, Discussion and Limitations
- Final term: `BSEI`
- Naming rule: use only `BSEI`; do not introduce or retain an expanded form
- Exact insertion/replacement: use `manuscript/sections/bsei.md`; replace the
  legacy method description only after preserving equation numbering
- Table: `manuscript/tables/bsei.csv` and `manuscript/tables/bsei.tex`
- Figure: `manuscript/figures/bsei.svg`
- Caption: `manuscript/captions/bsei.md`
- Dataset/split/checkpoint/seeds: not applicable
- Evidence: `models/convnext_pretrain.py`, `reports/architecture_inventory.json`
- Reviewer status: READY_FOR_TECHNICAL_REVIEW; not yet inserted in DOCX
- Remaining placeholders: Word equation/cross-reference numbering only

## WQ-002 — training objective and protocol

- Target: Sections 3.11 and 4.3
- Final text: protocol portion is `manuscript/sections/experiment.md`; objective
  equations and auxiliary weights are `manuscript/sections/loss.md`
- Objective table: `manuscript/tables/loss.csv` and `manuscript/tables/loss.tex`
- Objective figure: `manuscript/figures/loss.svg`
- Objective caption: `manuscript/captions/loss.md`
- Dataset/split: persisted manifests for Kvasir-SEG and CVC-ClinicDB
- Checkpoint: selected by equal mean validation Dice; test sets excluded
- Seeds: 42, 3407, 2026
- Evidence: `configs/training_protocol.yaml`, `research_pipeline/losses.py`, `train_research.py`
- Reviewer status: READY_FOR_TECHNICAL_REVIEW
- Remaining placeholders: Word equation numbering and cross-references only

## WQ-003 — numerical Results tables

- Target: Abstract and Section 5 tables/figures
- Status: BLOCKED_BY_EXPERIMENT
- Dataset/split: as defined in `configs/training_protocol.yaml`
- Checkpoint/seeds: official best checkpoint for seeds 42, 3407, 2026
- Evidence required: 18 completed official summaries, validated test reports and aggregate statistics
- Reviewer status: NOT_READY
- Remaining placeholders: all performance values, uncertainty, effect sizes and claims

## WQ-004 — fixed experimental protocol

- Target: Section 4.3, immediately after the opening implementation paragraph
- Final text: use `manuscript/sections/experiment.md`
- Table: `manuscript/tables/experiment.csv` and `manuscript/tables/experiment.tex`
- Figure: `manuscript/figures/experiment.svg`
- Caption: `manuscript/captions/experiment.md`
- Dataset/split: persisted Kvasir-SEG and CVC-ClinicDB development manifests
- Checkpoint: equal mean validation Dice; external test sets excluded from selection
- Seeds: 42, 3407, 2026
- Evidence: `configs/training_protocol.yaml`, `train_research.py`, `configs/splits/`
- Reviewer status: READY_FOR_TECHNICAL_REVIEW; not yet inserted in DOCX
- Remaining placeholders: none in the protocol text; performance results remain blocked by EXPERIMENT

## WQ-005 — architecture complexity

- Target: Sections 3.12 and 5.4; replace only the parameter/MAC/GFLOPs placeholders
- Final text: `manuscript/sections/complexity.md`
- Table: `manuscript/tables/complexity.csv` and `manuscript/tables/complexity.tex`
- Figure: `manuscript/figures/complexity.svg`
- Caption: `manuscript/captions/complexity.md`
- Dataset/split/checkpoint/seeds: not applicable; architecture-only measurement
- Evidence: `reports/architecture_gate.json`
- Method: input 1 x 3 x 352 x 352; GFLOPs = 2 x MACs
- Reviewer status: READY_FOR_TECHNICAL_REVIEW; not yet inserted in DOCX
- Remaining placeholders: latency, FPS and peak memory require a separate reproducible benchmark

## WQ-006 — special-control definitions

- Target: Methods ablation-protocol subsection and the future mechanism-isolation tables
- Exact insertion: add `manuscript/sections/controls.md` to the Methods ablation-protocol subsection; numerical comparison text remains blocked by official experiments
- Independent controls: 14; computation-identical controls are mapped through `control_reuse` and must not be reported as independently trained runs
- Coverage: backbone, BSEI alternatives, GDF fusion alternatives, MSC branch/dilation choices, detail-channel widths and deep-supervision head counts
- Dataset/split/checkpoint/seeds: official protocol and seeds 42, 3407, 2026; no control metric is currently manuscript eligible
- Evidence: `configs/ablation_matrix.yaml`; `reports/control_architecture_gate.json`; `pretrained/manifest.json`
- Table: `manuscript/tables/controls.csv` and `manuscript/tables/controls.tex`
- Figure: `manuscript/figures/controls.svg`
- Caption: `manuscript/captions/controls.md`
- Complexity table: `manuscript/tables/control_complexity.csv` and `manuscript/tables/control_complexity.tex`
- Complexity figure: `manuscript/figures/control_complexity.svg`
- Complexity caption: `manuscript/captions/control_complexity.md`
- Reviewer status: DEFINITIONS_READY; RESULTS_BLOCKED_BY_EXPERIMENT
- Remaining placeholders: every control performance value, uncertainty statistic, comparison claim, result table and result figure

## WQ-007 — limitations and evidence boundaries

- Target: Experimental Setup metrics/statistics paragraphs, Discussion limitations, and unresolved efficiency placeholders
- Exact insertion: use `manuscript/sections/limitations.md`; remove unsupported S-measure, latency, FPS, peak-memory, and patient-level leakage claims instead of supplying estimates
- Table: `manuscript/tables/reporting_decisions.csv` and `manuscript/tables/reporting_decisions.tex`
- Figure: `manuscript/figures/evidence_boundaries.svg`
- Caption: `manuscript/captions/limitations.md`
- Dataset/split: deterministic image-level manifests; patient, procedure, and source-video identifiers unavailable
- Checkpoint/seeds: official checkpoints; seeds 42, 3407, and 2026; descriptive sample SD and two-sided 95% Student-t interval with df=2
- Evidence: `configs/training_protocol.yaml`; `tools/aggregate_official.py`; `research_pipeline/evaluation.py`; `reports/dataset_file_audit.json`
- Reviewer status: READY_FOR_TECHNICAL_REVIEW; numerical result references remain blocked
- Remaining external-owner fields: code/archive URL and author contributions

## WQ-008 — exact DOCX paragraph actions

- Target map: `reports/docx_target_paragraphs.json`
- Inspection method: read-only extraction from `word/document.xml`; source DOCX remains unchanged
- READY_FOR_WORD: paragraphs 0, 50, 104, 117, 144, 163-166, 178, 193, and 229
- BLOCKED_BY_EXPERIMENT/EVALUATION: paragraphs 5, 234, 236, 243-245, and 254
- USER_REQUIRED: paragraphs 1-3, 279, and 285
- Deletion rule: unsupported efficiency values and S-measure text are deleted rather than filled with estimates
- Reviewer status: TARGETS_VERIFIED; application deferred until the controlled Word stage

## WQ-009 — author-owned fields

- Input checklist: `manuscript/user_inputs_required.md`
- Target: front matter and Declarations, paragraphs 1-3, 279, and 285
- Status: USER_REQUIRED; never infer or fabricate these fields
- Impact: does not block experiments; blocks placeholder-free final DOCX approval

## WQ-010 — qualitative prediction and error-map figures

- Target: qualitative Results subsection and associated figure captions
- Status: BLOCKED_BY_EXPERIMENT
- Selection protocol: deterministic dataset/size-bin selection using ground-truth area only; prediction performance is not used for case selection
- Prediction color: pink/magenta; error map uses TP white, FP yellow, FN cyan, and TN black
- Figure generator: `tools/generate_qualitative_figures.py`
- Evidence required: validated `full / seed 42` checkpoint and all five official evaluation outputs
- Final figures: `manuscript/figures/qualitative_<dataset>.png` after the gate passes
- Caption: `manuscript/captions/qualitative.md` after the gate passes
- Reviewer status: PROTOCOL_READY; FIGURES_NOT_READY

## WQ-011 — incremental and control numerical result packages

- Target: Abstract numerical claims and Section 5 result tables, plots, and discussion
- Status: BLOCKED_BY_EXPERIMENT
- Incremental generator: `tools/generate_incremental_result_artifacts.py`
- Control generator: `tools/generate_control_result_artifacts.py`
- Required evidence: six validated incremental aggregates, fourteen validated independent-control aggregates, and explicit provenance for eleven reused controls
- Planned tables: CSV and LaTeX files under `manuscript/tables/`
- Planned figures: evidence-derived SVG files under `manuscript/figures/`
- Planned text/captions: evidence-derived Markdown under `manuscript/sections/` and `manuscript/captions/`
- Reviewer status: GENERATORS_TESTED; NUMERICAL_CONTENT_NOT_READY

## WQ-012 — controlled DOCX readiness and render gate

- Pre-edit command: `python tools/check_word_update_readiness.py`
- Machine-readable evidence: `reports/word_update_readiness.json`
- Required pre-edit status: `READY`; any blocker forbids backup/edit execution
- Target-integrity contract: every paragraph index must be unique and in range;
  every action must be `replace` or `delete`; each ready replacement must have
  exactly one non-empty inline payload or an existing non-empty source file
- Plan command after readiness: `python tools/build_word_update_plan.py`; it
  strips Markdown section headings, preserves paragraph boundaries, records
  source hashes, and never writes a plan while the readiness gate is blocked
- Apply command after verified backup: `python tools/apply_word_update_plan.py`;
  it writes only a new DOCX, refuses source/output overwrite, verifies the
  backup manifest and hashes, checks table anchors and expected old cell values,
  applies run-local BSEI replacements, and rejects remaining legacy names or
  placeholders before retaining the output
- Source identity: SHA-256 must match `reports/docx_target_paragraphs.json`
- Author data contract: `manuscript/author_inputs.json` must contain all fields listed in `manuscript/user_inputs_required.md`
- Backup rule: create a timestamped byte-for-byte backup only after readiness is `READY` and before the first edit
- Post-edit render: export the edited DOCX to PDF, run `tools/render_pdf_winrt.ps1`, and inspect every rendered page
- Reviewer status: GATE_IMPLEMENTED_AND_TESTED; hash-bound plan/apply pipeline,
  mandatory backup, target/table integrity and full authorized regression PASS;
  LIVE_STATUS_BLOCKED
