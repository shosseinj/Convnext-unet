---

## description: Independently review validated pilot evidence and decide promotion

Act as an independent scientific evidence reviewer.

Repository:
C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet

Current verified state:

- Stage: PILOT_VALIDATION_PENDING
- Four isolated seed-42 pilot runs exist
- All 4/4 runs pass the strict technical validator
- No training or campaign process is active
- The next required gate is independent pilot-evidence review

Do not rerun training and do not merely repeat the existing validator.

Use only:

C:\Users\jafari.h\Desktop\ai_project.venv\Scripts\python.exe

Review the four pilot runs independently from the orchestration logic.

Required checks:

1. Identify the canonical four pilot variants and their exact output directories.

2. Confirm for every pilot:
   - variant identity
   - seed 42
   - isolated output namespace
   - expected epoch count
   - finite metrics
   - valid best and final checkpoints
   - complete history
   - complete summary
   - complete metadata
   - validator PASS
   - no overlap with official or legacy outputs

3. Compare logs, histories, summaries and checkpoints for contradictions.

4. Confirm that no pilot result was copied, reused, aliased or accidentally read from a legacy result directory.

5. Review the actual pilot metrics and determine whether they satisfy the repository’s documented pilot promotion criteria.

6. Do not invent promotion criteria. Read them from:
   - AGENTS.md
   - .agentic/state.json
   - workflow configuration
   - pilot protocol
   - validators
   - relevant project documentation

7. If no explicit numeric promotion criterion exists:
   - do not fabricate one
   - assess only technical validity and experimental readiness
   - record that scientific superiority is not yet established
   - promote only if the documented gate permits readiness-based promotion

8. Produce one concise evidence-review artifact:

   `reports/pilot_evidence_review.md`

It must contain only:

- Review scope
- Pilot matrix
- Evidence checked
- Per-run verdict
- Cross-run consistency
- Promotion criteria
- Gate decision
- Exact blockers, if any
- Recommended next stage

9. Update `.agentic/state.json` only after reaching an evidence-based decision.

10. Update `RUN_STATUS.md` without appending duplicate lines.

If the gate passes:

- Set the next stage exactly as defined by the workflow.
- Mark pilot evidence review PASS.
- Do not launch the next campaign in this command.
- Set `Next action` to launch the newly approved stage through `/resume-workflow`.

If the gate fails:

- Keep the stage at `PILOT_VALIDATION_PENDING`.
- Record the exact evidence deficiency.
- Do not rerun valid pilots unless the deficiency requires new training.

Do not perform manuscript, Word, control experiments, qualitative evaluation or full official training in this command.

Final response must contain only:

Stage:
Evidence review:
Pilots reviewed:
Gate:
Promoted to:
Blocker:
Report:
Next:
