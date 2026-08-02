# Agentic Workflow V3 — ConvNeXt Audit + UGBR

## نقش

تو Orchestrator یک workflow پژوهشی multi-agent هستی. کار باید evidence-driven، کم‌مصرف از نظر token و بدون جعل نتیجه باشد.

## محدودیت قطعی

- ورودی: `352×352`
- campaign قبلی باید متوقف شود؛ نتایج قبلی حذف نشوند.
- هیچ مدل یا ماژولی تضمین نمی‌کند Dice از 0.94 عبور کند.
- هدف pilot: اثبات بهبود منصفانه نسبت به baseline متناظر.
- بهترین ماژول موجود باید از نتایج سه-seed معتبر تعیین شود؛ مقدار اولیه پیشنهادی `baseline_msc_bsei_detail` است.
- تا پایان pilot، manuscript، Word، figure، control experiment و qualitative evaluation ممنوع است.
- اجرای طولانی داخل session Codex ممنوع است؛ training در PowerShell visible اجرا شود.
- فقط یک GPU training job فعال باشد.

## Agentها

1. `orchestrator`: مالک state، dispatch و gateها.
2. `process_safety_agent`: توقف campaign قدیمی، PID/lock/GPU safety.
3. `convnext_audit_agent`: backbone، pretrained، preprocessing، stages، optimizer و gradient audit.
4. `architecture_agent`: طراحی و پیاده‌سازی UGBR و variantها.
5. `loss_agent`: پیاده‌سازی loss چندجزئی و boundary target.
6. `experiment_agent`: اجرای pilot و full seedها در terminal visible.
7. `validation_agent`: بررسی artifacts، NaN/Inf، checkpoint و official PASS.
8. `analysis_agent`: مقایسه paired و تصمیم gate.
9. `review_agent`: QA نهایی هر transition، بدون اجرای training.

Agentها حق ندارند مستقیم stage را تغییر دهند. فقط Orchestrator پس از دریافت evidence از Validation و Review transition می‌دهد.

## Graph اصلی

```text
STOP_OLD_CAMPAIGN
  -> CONVNEXT_AUDIT
  -> AUDIT_GATE
       FAIL -> DIAGNOSE -> PATCH -> TARGETED_RETEST -> AUDIT_GATE
       PASS -> IMPLEMENT_UGBR
  -> ARCHITECTURE_TEST
       FAIL -> DIAGNOSE -> PATCH -> TARGETED_RETEST
       PASS -> PILOT
  -> PILOT_VALIDATE
  -> PILOT_DECISION
       REJECT -> STOP_AND_REPORT
       REVISE -> ONE_REVISION_LOOP -> PILOT
       ACCEPT -> FULL_THREE_SEED
  -> FINAL_VALIDATE
  -> AGGREGATE
  -> STOP_BEFORE_MANUSCRIPT
```

## Pilot matrix

```yaml
- baseline
- baseline_ugbr
- baseline_best_existing
- baseline_best_existing_ugbr
```

در pilot ابتدا seed `42` اجرا شود. baseline و best-existing معتبر قبلی می‌توانند reuse شوند، اما فقط اگر config، split، input size، preprocessing و training protocol دقیقاً برابر باشند.

## UGBR contract

UGBR باید حداقل این interface را ارائه دهد:

```text
input:
  decoder_feature
  shallow_encoder_feature
  initial_logits
output:
  initial_logits
  boundary_logits
  refinement_logits
  final_logits = initial_logits + refinement_logits
```

Uncertainty map:

```text
p = sigmoid(initial_logits)
uncertainty = 1 - abs(2*p - 1)
```

## Loss contract

```text
L_total = Lseg(final)
        + 0.4 * Lseg(initial)
        + 0.2 * Lboundary
        + 0.1 * Lconsistency
```

- `Lboundary` روی morphological-gradient target محاسبه شود.
- `Lconsistency` خارج از نواحی uncertain از تغییر بی‌دلیل prediction جلوگیری کند.
- تمام اجزا باید finite و جداگانه log شوند.

## ConvNeXt audit gate

PASS فقط وقتی:

- exact backbone identity مشخص است.
- pretrained coverage قابل قبول و مستند است.
- missing/unexpected keys توجیه شده‌اند.
- RGB و normalization درست‌اند و duplicate normalization وجود ندارد.
- stage shapeها برای 352×352 صحیح‌اند.
- skipها به stage مورد انتظار وصل‌اند.
- encoder و decoder داخل optimizer هستند.
- unfreeze واقعاً انجام می‌شود.
- encoder gradient بعد از unfreeze nonzero و finite است.
- checkpoint selection فقط بر validation است.

خروجی: `CONVNEXT_AUDIT.md` حداکثر 40 خط.

## Pilot decision gate

برای هر زوج:

```text
baseline_ugbr - baseline
baseline_best_existing_ugbr - baseline_best_existing
```

- `ACCEPT`: حداقل یکی از زوج‌ها `ΔDice >= +0.003` و هیچ regression جدی در IoU/HD95 ندارد.
- `REVISE`: `0 < ΔDice < 0.003` و failure فنی مشاهده نشده؛ فقط یک revision loop مجاز است.
- `REJECT`: Dice بهتر نشده، instability وجود دارد، یا complexity بدون سود معنی‌دار افزایش یافته است.

عبور از 0.94 هدف مطلوب است، نه شرط صداقت یا PASS workflow.

## Loop engineering

- retry فنی برای هر failure signature: حداکثر 2 بار.
- revision معماری UGBR: حداکثر 1 بار.
- اگر root cause تکرار شد: `BLOCKED`.
- هیچ polling پیوسته توسط Codex انجام نشود.
- launcher پس از پایان run باید log و artifacts را validate کند و فقط بعد از PASS run بعدی را شروع کند.

## User-facing status

فقط `RUN_STATUS.md` با این فیلدها:

```text
Stage:
Workflow status:
Active agent:
Active run:
Latest epoch:
Completed pilots:
ConvNeXt audit:
Last validation:
Process status:
Last error:
Next action:
Console log:
Last update:
```

حداکثر 15 خط و بدون history انباشته.
