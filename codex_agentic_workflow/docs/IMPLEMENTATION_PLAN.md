# ConvNeXt UGBR Agentic Workflow Implementation Plan

**Goal:** Audit ConvNeXt, add UGBR and composite loss, then run a gated four-variant pilot through a multi-agent graph.

## Tasks

1. Stop old campaign and verify zero repository GPU ownership.
2. Add state schema and initialize V3 state.
3. Implement ConvNeXt audit command and tests.
4. Implement UGBR module and architecture-factory variants with shape tests.
5. Implement boundary target, consistency loss and composite-loss tests.
6. Implement pilot campaign launcher with visible terminal, single log and official validation.
7. Run seed-42 pilot matrix and issue ACCEPT/REVISE/REJECT decision.
8. If ACCEPT, run three seeds and aggregate; otherwise stop cleanly.

Every implementation task must follow failing test -> minimal implementation -> targeted test -> regression test only when shared code changed.
