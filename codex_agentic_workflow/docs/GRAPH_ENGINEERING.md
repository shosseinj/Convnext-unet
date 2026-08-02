# Graph Engineering

## Design rules

- هر node یک مسئول و یک خروجی قابل اعتبارسنجی دارد.
- transition فقط با PASS مستقل Validation و Review انجام می‌شود.
- nodeهای GPU هم‌زمان نیستند.
- read-only auditها می‌توانند parallel باشند؛ writeها exclusive هستند.
- terminal nodeهای صریح: COMPLETE، STOP_AND_REPORT، BLOCKED.

## Mermaid

```mermaid
flowchart TD
  A[Stop old campaign] --> B[ConvNeXt audit]
  B --> C{Audit gate}
  C -- Fail --> D[Diagnose/Patch/Retest]
  D --> C
  C -- Pass --> E[Implement UGBR + Loss]
  E --> F{Architecture tests}
  F -- Fail --> G[Diagnose/Patch]
  G --> F
  F -- Pass --> H[Pilot seed 42]
  H --> I[Independent validation]
  I --> J{Pilot decision}
  J -- Reject --> K[Stop and report]
  J -- Revise once --> E
  J -- Accept --> L[Three-seed runs]
  L --> M[Final validation]
  M --> N[Aggregate and stop before manuscript]
```
