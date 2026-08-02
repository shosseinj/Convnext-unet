# Article Completion Monitoring Dashboard

این فایل داشبورد اصلی پروژه است. Codex باید پایان هر مرحله آن را با evidence به‌روزرسانی کند.

## وضعیت کلی

- Project: `Convnext-unet-main`
- Article: `hossein_paper_revised.docx`
- Official module: `BSEI`
- Current stage: `EXPERIMENT`
- Overall status: `IN_PROGRESS`
- Last update: `2026-08-01 10:52 UTC`
- Blocker: `None`; Word visual render remains unresolved but does not block the active experiment

## وضعیت‌ها

- `[ ]` انجام نشده
- `[~]` در حال انجام
- `[x]` انجام شده و evidence دارد
- `[!]` blocked یا نیازمند تصمیم کاربر

## Phase 1 — Audit

- [x] ثبت Git commit و environment
- [x] inventory مدل‌ها و train/evaluation scripts
- [x] inventory مقاله: sections، tables، figures و placeholderها
- [~] یکسان‌سازی نام BSEI در کد و مقاله (repository text پاک است؛ DOCX در stage Word باقی مانده)
- [x] تأیید implementation واقعی BSEI
- [x] تولید `reports/repository_audit.md`
- [x] تولید `reports/manuscript_inventory.md`

## Phase 2 — Protocol

- [x] تأیید split train/validation/test
- [x] تأیید سه seed
- [x] تثبیت loss، LR، batch، augmentation و epochs
- [x] تثبیت checkpoint selection با validation Dice
- [ ] تکمیل Materials and Methods
- [x] تولید جدول protocol و reproducibility

## Phase 3 — Implementation

- [x] configurable baseline
- [x] فعال‌سازی MSC
- [x] فعال‌سازی BSEI
- [x] فعال‌سازی Detail Branch
- [x] فعال‌سازی GDF
- [x] فعال‌سازی auxiliary heads برای DS
- [x] backbone controls
- [x] architecture tests
- [ ] شکل معماری نهایی و caption

## Phase 4 — Experiments

- [x] smoke test
- [x] pilot seed 42
- [~] incremental ablation با سه seed
- [ ] BSEI mechanism ablation
- [ ] GDF fusion ablation
- [ ] MSC branch/dilation ablation
- [ ] detail channel ablation
- [ ] deep supervision head ablation
- [ ] backbone comparison
- [~] raw logs/checkpoints/metadata

## Phase 5 — Statistics and Figures

- [ ] mean±std بین seedها
- [x] Params و GFLOPs
- [ ] Dice/IoU بر اساس اندازه پولیپ
- [ ] جدول incremental
- [ ] جدول mechanism isolation
- [ ] جدول backbone
- [x] جدول complexity
- [ ] شکل معماری
- [ ] شکل ablation trend
- [ ] شکل size-stratified analysis
- [ ] qualitative figure شامل input، ground truth، prediction و error map

## Phase 6 — Word Article

- [ ] Abstract
- [ ] Introduction و Contributions
- [ ] Related Work و citation audit
- [ ] Materials and Methods
- [ ] equations و channel counts
- [ ] Experimental Protocol
- [ ] جدول نتایج پنج دیتاست
- [ ] جدول cumulative ablation
- [ ] جدول mechanism isolation
- [ ] جدول complexity
- [ ] Figures 1–9 یا حذف placeholderهای غیرضروری
- [ ] Discussion
- [ ] Limitations
- [ ] Declarations، Data/Code availability و AI statement
- [ ] حذف همه `XX`، `TBD` و placeholderهای بدون owner

## Phase 7 — Final QA

- [ ] consistency متن/جدول/شکل
- [ ] numeric claim audit
- [ ] protocol leakage audit
- [ ] reference/DOI audit
- [ ] Word render و visual QA
- [ ] backup نسخه قبلی Word
- [ ] تحویل DOCX و PDF نهایی

## Failure ledger

| ID | Stage | Error | Root cause | Action | Attempts | Status |
|---|---|---|---|---|---:|---|
| F-001 | AUDIT | DOCX packaged render failed: `ModuleNotFoundError` for `pdf2image` and `docx` in project venv | Required document packages are absent from the selected project runtime; LibreOffice/Poppler are not in PATH | Used installed Word for read-only PDF export and the dependency-free Windows.Data.Pdf API for faithful PNG rasterization | 2 | Resolved; 29/29 source pages visually inspected |
| F-002 | TEST | Targeted unittest import failed for `tests.test_*` | `tests` is not a Python package | Re-ran the same three files with filename-based unittest discovery; 8/8 passed | 1 | Resolved |
| F-003 | AUDIT | `tests/test_video_evaluation.py` exists outside the permitted workflow scope | Artifact predates current continuation and conflicts with the explicit exclusion of Video QA Tester | Do not execute it; retain only until user-owned unrelated changes can be safely separated | 1 | Contained |
| F-004 | EXPERIMENT | Run metadata omitted environment versions, parameter count, complexity convention and source fingerprints | Initial runner recorded protocol/seed/split/commit but not the full evidence contract | Captured the active-run fingerprint and added deterministic evidence metadata for subsequent runs | 1 | Resolved |
| F-005 | EXPERIMENT | Metadata retest used a Windows backslash manifest key | `Path.relative_to()` was stringified without portable normalization | Changed manifest key serialization to POSIX form and reran targeted plus regression tests | 1 | Resolved |
| F-006 | EXPERIMENT | Resume did not preserve RNG/DataLoader trajectory or validate immutable run identity | RNG states were absent and resume compared only variant plus manifest hash | Added RNG/generator checkpoint restore, pretrained-weight fingerprint and immutable metadata validation; 2 targeted and 8 regression tests passed | 1 | Resolved for future checkpoints; active run is uninterrupted-only |
| F-007 | EXPERIMENT | Active baseline checkpoint predates the reproducible recovery contract | Run began before F-006 patch and cannot be made trajectory-equivalent after interruption | Do not interrupt; accept only if it completes uninterrupted, otherwise preserve failure evidence and restart from epoch 0 under the fixed contract | 1 | Contained |
| F-008 | EXPERIMENT | Queue trusted `summary.completed`, had no lock and logged incomplete recovery context | Launcher lacked centralized artifact validation and exclusive ownership | Added completion validation, finite-metric checks, structured events and a PID lock for safe future relaunch | 1 | Resolved |
| F-009 | TEST | Queue-lock test closed its own child process output twice | POSIX-style `os.kill(pid, 0)` is unsafe for this Windows runtime | Replaced Windows liveness check with `OpenProcess`; targeted queue/recovery tests and 8 regressions passed | 2 | Resolved |
| F-010 | EXPERIMENT | Early-stopping patience counted frozen-encoder epochs and resumed timing was segment-only | Runner incremented patience regardless of freeze phase and did not persist cumulative elapsed time | Patience now starts after unfreeze; cumulative elapsed time is checkpointed; targeted recovery tests passed | 1 | Resolved for future runs |
| F-011 | EXPERIMENT | Runtime and workflow ablation matrices could drift | Two YAML files encoded different schemas and queue used hard-coded jobs | Designated `configs/ablation_matrix.yaml` canonical, converted workflow copy to pointer, and derived 18 queue jobs from canonical YAML | 1 | Resolved for incremental matrix |
| F-012 | TEST | Architecture-factory test expected 15 independent controls after a duplicate control was converted to reuse | Test count was stale after `control_gdf_concatenation` was mapped to the computation-identical `baseline_msc_bsei_detail` result | Updated the expected independent-control count to 14 and reran all authorized regressions | 1 | Resolved; 21/21 PASS |
| F-013 | TEST | Control-queue test could not import its sibling official-queue module | Direct-script and package-import contexts expose different module roots | Added a package-relative import with direct-script fallback, then reran targeted and authorized regressions | 1 | Resolved; 23/23 PASS |
| F-014 | WORD_UPDATE | Control-complexity generator failed Python compilation | A LaTeX backslash was embedded in an f-string expression | Precomputed the escaped label outside the expression; compile, generation, SVG parse and source reconciliation then passed | 1 | Resolved |
| F-015 | WORD_UPDATE | First read-only DOCX paragraph extraction command failed PowerShell parsing | A pipeline followed a `foreach` block without grouping | Assigned the loop output to a variable before JSON conversion; 19 target paragraphs were then extracted without modifying the DOCX | 1 | Resolved |
| F-016 | WORD_UPDATE | In-app browser could not open the local PDF | Browser-control transport closed during setup | Used local Office render proof instead; no visual PASS inferred from the failed browser call | 1 | Contained; final QA still required |
| F-017 | WORD_UPDATE | First full-render invocation could not resolve the short `powershell.exe` name | The shell environment did not expose that executable by short name | Retried with the explicit system PowerShell path | 1 | Resolved |
| F-018 | WORD_UPDATE | Automated 29-page Office render loops hung before producing output | Clipboard-based range copying and in-loop PDF export were unstable under headless Office automation | Replaced clipboard rasterization with Windows.Data.Pdf direct PDF-page rendering; retained Word only for controlled PDF export and preserved the user's Word process | 4 | Resolved for render path; final edited DOCX still requires a fresh full QA pass |
| F-019 | TEST | Control-result artifact test expected 120 rows although the canonical matrix defines 25 displayed controls over five datasets | The test expectation omitted one valid displayed control; the generator correctly produced 125 rows | Corrected the stale assertion and reran the full authorized suite | 1 | Resolved; 35/35 PASS |
| F-020 | WORD_UPDATE | Word-ready artifacts introduced inconsistent expanded forms for the official BSEI name | Earlier drafting inferred two different expansions although the project contract requires the official name to be only `BSEI` | Removed all expanded forms from text/code artifacts, corrected the DOCX title target, reran the terminology scan and manuscript audit | 1 | Resolved; zero expanded-form or legacy-name hits in manuscript artifacts |
| F-021 | TEST | First full-regression counting wrapper stopped on a non-fatal `timm` FutureWarning | PowerShell promoted native stderr to `NativeCommandError` under `ErrorActionPreference=Stop` before unittest exit codes could be evaluated | Re-ran the same filename-scoped suite with native stderr tolerated and explicit per-process exit-code checks | 1 | Resolved; 38/38 authorized tests PASS |
| F-022 | WORD_UPDATE | A copied author-input template with nested empty values could satisfy a shallow top-level nonempty check | Lists and dictionaries were previously considered complete based only on container length | Made author-input validation recursive and added a nested-empty regression test | 1 | Resolved; empty names, mappings, URLs and role arrays remain blocked |
| F-023 | WORD_UPDATE | Paragraph 50 mapped an entire CSV into a caption and the target map did not guarantee removal of every legacy module name | The first map modeled only direct body paragraphs and conflated the adjacent Word table with its caption | Split caption text from an expected-value table-cell patch, added document-wide BSEI terminology replacements, and enforced structured/global update validation | 1 | Resolved in plan/applicator contract; final DOCX still requires live render QA |

## Evidence ledger

| Date | Stage | Artifact | Command/test | Evidence path | Reviewer |
|---|---|---|---|---|---|
| 2026-08-01 | AUDIT | Repository and architecture inventories | Git/code/data inspection | `reports/repository_audit.md`; `reports/architecture_inventory.json` | Codex verified |
| 2026-08-01 | AUDIT | Manuscript OOXML inventory and placeholder ledger | read-only OOXML inspection | `reports/manuscript_inventory.md`; `reports/manuscript_placeholders.csv` | Visual review pending |
| 2026-08-01 | IMPLEMENT | Architecture gate | `tools/validate_architectures.py` evidence review | `reports/architecture_gate.json` | Codex verified |
| 2026-08-01 | TEST | Targeted unit/regression tests | `python -m unittest discover -s tests -p <approved test file> -v` | console log in active Codex run; 8 tests passed | Codex verified |
| 2026-08-01 | TEST | Smoke gate | evidence/schema review | `reports/smoke_gate.json`; `results/raw/*/seed_42/smoke/` | Codex verified |
| 2026-08-01 | PILOT | Six-variant seed-42 pilot | aggregate/schema review | `results/aggregated/pilot_seed42.*`; `results/raw/*/seed_42/pilot/` | Not manuscript eligible |
| 2026-08-01 | EXPERIMENT | Official queue active; baseline seed 42 reached epoch 43 | process/log/history inspection | `results/raw/official_queue.jsonl`; `results/raw/baseline/seed_42/official/` | In progress |
| 2026-08-01 | EXPERIMENT | Active-run source/environment fingerprint | SHA-256 and runtime version capture before metadata-only patch | `results/raw/baseline/seed_42/official/active_run_source_fingerprint.json` | Codex verified |
| 2026-08-01 | EXPERIMENT | Extended run-evidence metadata | `py_compile`; metadata targeted test; 8 approved regression tests | `train_research.py`; console evidence in active Codex run | PASS for future run launches |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 66 | process/history/checkpoint inspection | `results/raw/baseline/seed_42/official/history.json`; `best.pth`; `last.pth` | In progress; 0/18 complete |
| 2026-08-01 | EXPERIMENT | Reproducible recovery contract | `test_recovery_metadata.py`; approved regression suite | `train_research.py`; `tests/test_recovery_metadata.py` | 10/10 tests PASS |
| 2026-08-01 | EXPERIMENT | Specialized-agent live health audit | process delta, GPU utilization, ZIP integrity, CPU checkpoint load | `results/raw/baseline/seed_42/official/`; queue logs | ACTIVE/HEALTHY; uninterrupted-only |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 76 | process/history inspection | `results/raw/baseline/seed_42/official/history.json` | In progress; 0/18 complete |
| 2026-08-01 | EXPERIMENT | Queue completion/relaunch hardening | 2 queue tests, 2 recovery tests, 8 approved regression tests | `tools/run_official_queue.py`; `tests/test_official_queue.py`; `.agentic/official_queue.lock.json` | 12/12 PASS |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 80 | process/history inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE/HEALTHY; 0/18 complete |
| 2026-08-01 | EXPERIMENT | Official evaluation and aggregation guards | metric tests, incomplete-run guard, three-seed statistics tests | `tools/evaluate_official.py`; `tools/aggregate_official.py`; related tests | PASS; no incomplete result promoted |
| 2026-08-01 | EXPERIMENT | Canonical incremental matrix | queue/factory tests | `configs/ablation_matrix.yaml`; workflow pointer; `tools/run_official_queue.py` | 18 jobs verified |
| 2026-08-01 | WORD_UPDATE | Complexity artifacts from passed architecture gate | generator plus CSV/SVG source reconciliation | `manuscript/{sections,tables,figures,captions}/complexity.*` | READY_FOR_TECHNICAL_REVIEW |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 86 | process/history inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE/HEALTHY; 0/18 complete |
| 2026-08-01 | WORD_UPDATE | BSEI code-to-equation/channel mapping | four-level tensor introspection and depthwise-group/padding checks | `manuscript/{sections,tables,figures,captions}/bsei.*`; `models/convnext_pretrain.py` | READY_FOR_TECHNICAL_REVIEW |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 88 | process/history inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE/HEALTHY; 0/18 complete |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 89 | process/history inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE/HEALTHY; 0/18 complete |
| 2026-08-01 | EXPERIMENT | Special-control implementation and local backbone weights | strict torchvision weight load; config/factory inspection | `configs/ablation_matrix.yaml`; `pretrained/manifest.json`; `models/convnext_pretrain.py` | Codex verified; no control result claimed |
| 2026-08-01 | TEST | Authorized regression suite after control deduplication | eight filename-scoped unittest discovery commands | `tests/test_*.py` excluding `test_video_evaluation.py` | 21/21 PASS; Video QA Tester not executed |
| 2026-08-01 | TEST | Legacy pilot checkpoint compatibility | CPU strict-load of baseline and full pilot `best.pth` | `results/raw/{baseline,full}/seed_42/pilot/best.pth` | PASS; zero missing/unexpected keys |
| 2026-08-01 | IMPLEMENT | Fourteen independent special-control architecture smoke gates | `python tools/validate_architectures.py --sections controls --input-size 64 --device cpu --output reports/control_architecture_smoke_gate.json` | `reports/control_architecture_smoke_gate.json` | PASS; architecture evidence only |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 105 | process/history/checkpoint inspection | `results/raw/baseline/seed_42/official/history.json`; `best.pth`; `last.pth` | ACTIVE/HEALTHY; 0/18 complete |
| 2026-08-01 | EXPERIMENT | Gated special-control queue | control job derivation and missing-incremental prerequisite test | `tools/run_control_queue.py`; `tests/test_control_queue.py` | 42 unique jobs prepared; launch correctly blocked before 18-run gate |
| 2026-08-01 | TEST | Authorized regression suite after control-queue import fix | nine filename-scoped unittest discovery commands | approved tests excluding `test_video_evaluation.py` | 23/23 PASS; Video QA Tester not executed |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 108 | process/history inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE/HEALTHY; 0/18 complete |
| 2026-08-01 | WORD_UPDATE | Special-control protocol artifacts | CSV family count, SVG XML parse and canonical-ID reconciliation | `manuscript/{sections,tables,figures,captions}/controls.*`; WQ-006 | DEFINITIONS_READY; all numerical results blocked |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 109 | process/GPU/history/checkpoint/fatal-signature inspection | `results/raw/baseline/seed_42/official/`; `results/raw/official_queue.jsonl` | ACTIVE/HEALTHY; 0/18 complete |
| 2026-08-01 | VALIDATE | Incremental finalization orchestrator and hard completion guard | 2 targeted tests, full authorized regression, live `--check-only` guard | `tools/finalize_incremental.py`; `tests/test_finalize_incremental.py` | 25/25 tests PASS; live finalization correctly blocked at 0/18 |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 113 | history/completion inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE; summary absent; 0/18 complete |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 115 | process/history/completion inspection | `results/raw/baseline/seed_42/official/history.json` | Four processes active; summary absent; 0/18 complete |
| 2026-08-01 | IMPLEMENT | Full-resolution control architecture gate | `validate_architectures.py --sections controls --input-size 352 --device cpu` | `reports/control_architecture_gate.json` | PASS for all 14 independent controls |
| 2026-08-01 | WORD_UPDATE | Control complexity artifacts | compile, generator, CSV-to-gate reconciliation and SVG XML parse | `manuscript/tables/control_complexity.*`; `manuscript/figures/control_complexity.svg`; caption | PASS; architecture cost only |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 117 | history/completion inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE; summary absent; 0/18 complete |
| 2026-08-01 | AGGREGATE | Hash-bound aggregate provenance | targeted source-hash test plus full authorized regression | `tools/aggregate_official.py`; `tests/test_official_aggregation.py` | 26/26 PASS; future aggregates record status, method, timestamp and 15 source hashes |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 121 | history/completion inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE; summary absent; 0/18 complete |
| 2026-08-01 | WORD_UPDATE | Guarded incremental result-artifact generator | compile, missing-aggregate guard and full authorized regression | `tools/generate_incremental_result_artifacts.py`; related test | 27/27 PASS; no incomplete numerical artifact produced |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 124 | history/completion inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE; summary absent; 0/18 complete |
| 2026-08-01 | TEST | Synthetic end-to-end result-artifact generation | in-memory six-variant/five-dataset fixture, CSV row count and SVG parse | `tests/test_incremental_result_artifacts.py` | PASS; 28 unique authorized tests now covered |
| 2026-08-01 | AUDIT | Repository-text terminology scan | `rg` legacy-name scan excluding binary DOCX | `reports/architecture_inventory.json`; repository text | PASS; only BSEI remains in text repository; DOCX pending controlled update |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 126 | history/completion inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE; summary absent; 0/18 complete |
| 2026-08-01 | VALIDATE | Gated control finalization orchestrator | compile, targeted 42-run guard test and live `--check-only` | `tools/finalize_controls.py`; `tests/test_finalize_controls.py` | PASS; correctly blocked at 0/42; 29 unique tests covered |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 42 reached epoch 127 | history/completion inspection | `results/raw/baseline/seed_42/official/history.json` | ACTIVE; summary absent; 0/18 complete |
| 2026-08-01 | EXPERIMENT | Baseline seed 42 genuine completion | queue finish code, 150-row history, summary and checkpoint integrity | `results/raw/baseline/seed_42/official/{summary.json,history.json,best.pth,last.pth}` | PASS; 1/18 complete |
| 2026-08-01 | VALIDATE | Baseline seed 42 deep run validation | `python tools/validate_official_run.py --variant baseline --seed 42` | `results/raw/baseline/seed_42/official/validation.json` | PASS; legacy-uninterrupted contract; artifacts hash-bound |
| 2026-08-01 | TEST | Post-run authorized regression suite | twelve filename-scoped unittest discovery commands | approved tests excluding Video QA Tester | 29/29 PASS |
| 2026-08-01 | EXPERIMENT | Official queue transition to baseline seed 3407 epoch 1 | process, queue, config metadata and history inspection | `results/raw/official_queue.jsonl`; `results/raw/baseline/seed_3407/official/` | ACTIVE/HEALTHY; extended evidence metadata present; 1/18 complete |
| 2026-08-01 | VALIDATE | Baseline seed 42 evaluation scheduling | compute-isolation review | `.agentic/state.json` evaluation backlog | Deep validation PASS; five-dataset evaluation deferred until the official GPU queue is idle to avoid resource competition |
| 2026-08-01 | EXPERIMENT | Persistent experiment campaign guardian | compile, PID-lock/ledger inspection, targeted test and full authorized regression | `tools/run_experiment_campaign.py`; `.agentic/experiment_campaign.lock.json`; `results/raw/experiment_campaign.jsonl` | PID 35652 ACTIVE; 1/18 validated; 30/30 tests PASS |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 3407 reached epoch 4 | process/history inspection | `results/raw/baseline/seed_3407/official/` | ACTIVE/HEALTHY; 1/18 complete |
| 2026-08-01 | WORD_UPDATE | Limitations and evidence-boundary package | CSV row validation, SVG XML parse, legacy-term scan and source inspection | `manuscript/sections/limitations.md`; reporting-decision table; evidence-boundary figure/caption; WQ-007 | READY_FOR_TECHNICAL_REVIEW; external author fields pending |
| 2026-08-01 | REVIEW | Non-numerical manuscript artifact gate | fail-closed artifact audit across seven packages | `reports/manuscript_artifact_gate.json`; `tools/audit_manuscript_artifacts.py` | PASS for 7 packages; numerical results correctly BLOCKED_BY_EXPERIMENT; 31 tests covered |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 3407 reached epoch 8 | campaign/process/history inspection | `results/raw/baseline/seed_3407/official/`; campaign ledger | ACTIVE/HEALTHY; guardian active; 1/18 complete |
| 2026-08-01 | EXPERIMENT | Baseline seed 3407 epoch-10 evidence audit | metadata, checkpoint ZIP, history, process/GPU and fatal-signature inspection | `results/raw/baseline/seed_3407/official/` | ACTIVE/HEALTHY; extended evidence contract verified |
| 2026-08-01 | WORD_UPDATE | Exact source-DOCX paragraph action map | read-only OOXML extraction plus source SHA-256 | `reports/docx_target_paragraphs.json`; WQ-008 | 24 targets mapped: 12 ready, 7 blocked, 5 user-required; DOCX unchanged |
| 2026-08-01 | WORD_UPDATE | Source-DOCX render-path proof | Word read-only PDF export, page-1 EMF/PNG export and visual inspection | `reports/docx_source_render.pdf`; `reports/docx_source_render_pages/page-001.png`; render proof JSON | PARTIAL PASS only; 29-page final visual QA remains required |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 3407 reached epoch 26 | campaign/process/history inspection | `results/raw/baseline/seed_3407/official/`; campaign ledger | ACTIVE/HEALTHY; campaign guardian active; 1/18 complete |
| 2026-08-01 | WORD_UPDATE | Author-owned input checklist | DOCX front-matter/declaration mapping | `manuscript/user_inputs_required.md`; WQ-009 | Five factual fields identified; experiments unaffected; final placeholder-free DOCX depends on user/author input |
| 2026-08-01 | EXPERIMENT | Official queue health; baseline seed 3407 reached epoch 27 | state/history inspection | `results/raw/baseline/seed_3407/official/history.json` | ACTIVE; guardian active; 1/18 complete |

| 2026-08-01 | EXPERIMENT | Current campaign reconciliation | process command lines, PID lock, official summary validation and active history inspection | `.agentic/experiment_campaign.lock.json`; `results/raw/baseline/seed_3407/official/history.json`; `results/raw/baseline/seed_42/official/validation.json` | Guardian PID 28016 ACTIVE; baseline/3407 epoch 36; 1/18 official complete |
| 2026-08-01 | TEST | Full authorized regression after manuscript artifact audit hardening | 16 filename-scoped unittest discovery commands, excluding `test_video_evaluation.py` | `tests/test_*.py` except excluded Video QA test | 35/35 PASS; Video QA Tester not executed |
| 2026-08-01 | WORD_UPDATE | Manuscript artifact gate including conditional numerical-package checks | `python tools/audit_manuscript_artifacts.py` | `reports/manuscript_artifact_gate.json` | PASS for 7 nonnumerical packages; numerical results BLOCKED_BY_EXPERIMENT |

| 2026-08-01 | WORD_UPDATE | Faithful full-source render path | Word read-only PDF export; `tools/render_pdf_winrt.ps1`; visual inspection of every PNG | `reports/docx_source_render.pdf`; `reports/docx_source_render_winrt/render_manifest.json`; `reports/docx_source_render_proof.json` | PASS for 29/29 source pages; final edited DOCX still pending |
| 2026-08-01 | EXPERIMENT | Concurrent queue health during render work | active history, guardian PID lock and official-summary inspection | `results/raw/baseline/seed_3407/official/history.json`; `.agentic/experiment_campaign.lock.json` | baseline/3407 epoch 47; guardian PID 28016 ACTIVE; 1/18 complete |

| 2026-08-01 | EXPERIMENT | baseline seed 3407 epoch-50 checkpoint health gate | ZIP integrity, CPU `torch.load`, metadata/RNG inspection, finite-history and PID/fatal-signature checks | `results/raw/baseline/seed_3407/official/{history.json,best.pth,last.pth,resolved_config.json}` | PASS as progress evidence; best and last load; last contains RNG/cumulative elapsed; run advanced to epoch 51; 1/18 complete |

| 2026-08-01 | WORD_UPDATE | Strict BSEI-only naming gate | repository expanded-form scan, manuscript legacy-name scan, JSON parse, artifact audit and targeted unittest | `manuscript/sections/bsei.md`; `manuscript/captions/bsei.md`; `manuscript/word_update_queue.md`; `reports/docx_target_paragraphs.json` | PASS; only `BSEI` retained; numerical packages remain blocked by experiment |
| 2026-08-01 | EXPERIMENT | Queue continued during terminology correction | active-history inspection | `results/raw/baseline/seed_3407/official/history.json` | ACTIVE at epoch 54; 1/18 complete |

| 2026-08-01 | WORD_UPDATE | Fail-closed final DOCX readiness gate | compile; 3 targeted tests; live gate with expected exit 2 | `tools/check_word_update_readiness.py`; `tests/test_word_update_readiness.py`; `reports/word_update_readiness.json` | BLOCKED correctly: unresolved experiment/evaluation targets and author inputs; source hash matches |
| 2026-08-01 | TEST | Full authorized regression after readiness gate | 17 filename-scoped unittest commands with explicit exit-code checks | `tests/test_*.py` excluding `test_video_evaluation.py` | 38/38 PASS; Video QA Tester not executed |
| 2026-08-01 | EXPERIMENT | Training continued during CPU-only gate tests | active-history inspection | `results/raw/baseline/seed_3407/official/history.json` | ACTIVE at epoch 57; 1/18 complete |

| 2026-08-01 | WORD_UPDATE | Verified pre-edit backup contract and structured author-input template | compile, blocked-live-run check, 3 backup tests and recursive author-input tests | `manuscript/author_inputs.template.json`; `tools/prepare_word_update_backup.py`; `tests/test_word_update_backup.py` | PASS; live readiness created no backup; source DOCX unchanged |
| 2026-08-01 | TEST | Full authorized regression after backup/readiness hardening | 18 filename-scoped unittest commands with explicit exit-code checks | `tests/test_*.py` excluding `test_video_evaluation.py` | 42/42 PASS; Video QA Tester not executed |
| 2026-08-01 | EXPERIMENT | Training continued during Word tooling tests | active-history and guardian inspection | `results/raw/baseline/seed_3407/official/history.json`; `.agentic/experiment_campaign.lock.json` | ACTIVE at epoch 61; guardian PID 28016 alive; 1/18 complete |
| 2026-08-01 | EXPERIMENT | Live campaign health reconciliation | local plus specialized-agent history/process/checkpoint-ZIP/completion-marker/fatal-signature inspection | `results/raw/baseline/seed_3407/official/{resolved_config.json,history.json,best.pth,last.pth}`; `.agentic/experiment_campaign.lock.json`; `results/raw/{official_queue,experiment_campaign}.jsonl` | ACTIVE/HEALTHY at epoch 65; queue/train/guardian chains alive; checkpoint ZIPs valid; no summary/result/completed marker yet; no fatal/OOM/NaN; 1/18 validated |
| 2026-08-01 | WORD_UPDATE | Fail-closed paragraph-target integrity gate | six targeted tests, live readiness check, and full authorized regression | `tools/check_word_update_readiness.py`; `tests/test_word_update_readiness.py`; `reports/word_update_readiness.json`; WQ-012 | PASS: duplicate/out-of-range targets, invalid actions, and missing/ambiguous replacement payloads are rejected; live gate remains BLOCKED without editing |
| 2026-08-01 | TEST | Full authorized regression after Word target-integrity hardening | 18 filename-scoped unittest commands with explicit exit checks | `tests/test_*.py` excluding `test_video_evaluation.py` | 44/44 PASS; Video QA Tester not executed |
| 2026-08-01 | EXPERIMENT | Training continued during CPU-only Word gate regression | history/guardian/completion-marker/fatal-signature inspection | `results/raw/baseline/seed_3407/official/history.json`; `.agentic/experiment_campaign.lock.json` | ACTIVE/HEALTHY at epoch 68; guardian alive; no summary yet; fatal scan clean; 1/18 validated |
| 2026-08-01 | WORD_UPDATE | Hash-bound Word update-plan materializer | seven readiness tests, two plan tests, blocked live execution, full authorized regression | `tools/build_word_update_plan.py`; `tests/test_word_update_plan.py`; `reports/docx_target_paragraphs.json`; WQ-012 | PASS: explicit markdown/source-reference modes; live plan correctly not created while gate is BLOCKED |
| 2026-08-01 | TEST | Full authorized regression after Word plan implementation | 19 filename-scoped unittest commands with explicit exit checks | `tests/test_*.py` excluding `test_video_evaluation.py` | 47/47 PASS; Video QA Tester not executed |
| 2026-08-01 | EXPERIMENT | Training continued during CPU-only Word plan tests | history/guardian/completion-marker inspection | `results/raw/baseline/seed_3407/official/history.json`; `.agentic/experiment_campaign.lock.json` | ACTIVE at epoch 72; guardian alive; no summary yet; 1/18 validated |
| 2026-08-01 | WORD_UPDATE | Deterministic OOXML Word applicator with mandatory backup | fixture DOCX apply, ZIP/source-hash checks, table-anchor drift test path, legacy/placeholder post-scan and blocked live execution | `tools/apply_word_update_plan.py`; `tests/test_apply_word_update_plan.py`; `reports/docx_target_paragraphs.json` | PASS on fixtures; source preserved; live output correctly not created while readiness is BLOCKED and no backup exists |
| 2026-08-01 | TEST | Full authorized regression after controlled Word applicator | 20 filename-scoped unittest commands with explicit exit checks | `tests/test_*.py` excluding `test_video_evaluation.py` | 51/51 PASS; Video QA Tester not executed |
| 2026-08-01 | EXPERIMENT | Training continued during CPU-only OOXML tests | history/guardian/completion/fatal-signature inspection | `results/raw/baseline/seed_3407/official/history.json`; `.agentic/experiment_campaign.lock.json` | ACTIVE/HEALTHY at epoch 78; guardian alive; no summary; fatal scan clean; 1/18 validated |

## Decisions required

- [x] تطبیق معادلات آماده درج با رفتار channel-wise و channel counts واقعی BSEI؛ درج Word در stage مربوطه باقی مانده است
- [x] split رسمی development برای سه seed در manifestها تثبیت شد؛ نبود patient/video identifiers همچنان باید به‌عنوان محدودیت گزارش شود
- [!] تکمیل author names، affiliations و corresponding email
- [!] تعیین تکلیف مقایسه‌های خارجی فاقد protocol قابل‌بازسازی
