---
description: Reconcile current Codex changes, then start the gated ConvNeXt/UGBR workflow
agent: orchestrator
---

Start or continue the workflow in `agentic_workflow/configs/graph.yaml`.

Mandatory first node: RECONCILE_CURRENT_REPO. Treat the current repository as source of truth. Preserve correct Codex changes and existing evidence. Do not blindly copy bundled designs into source code.

Then proceed through process safety, ConvNeXt audit, implementation gap closure, seed-42 pilots, validation, and promotion gates. Long training must run in a visible PowerShell terminal and log to CAMPAIGN_CONSOLE.log. Exit the interactive OpenCode turn after healthy launch; do not poll epochs continuously.
