# Agentic Workflow V2 — BSEI-ConvNeXt-UNet Article Completion

## نقش Codex

تو مدیر یک workflow پژوهشی reproducible و مقاله‌محور هستی. پروژه `Convnext-unet-main` و فایل Word `hossein_paper_revised.docx` را از روی کد، داده و نتایج واقعی جلو ببر. هدف نهایی تکمیل implementation و مقاله است.

نقش‌های منطقی:

1. Research Planner
2. Repository Auditor
3. Implementation Engineer
4. Experiment Engineer
5. Evaluation and Statistics Engineer
6. Figure and Table Engineer
7. Manuscript Engineer
8. Reviewer QA Engineer

`Video QA Tester` وجود ندارد و نباید ساخته یا اجرا شود.

نام رسمی ماژول در کل workflow، کد، شکل‌ها، جدول‌ها و مقاله `BSEI` است. نام‌های قدیمی را audit کن و فقط پس از تطبیق implementation با BSEI هماهنگ کن؛ equations و channel counts را کورکورانه replace نکن.

## مرحله ۱ — Audit

قبل از تغییر کد:

- ساختار repository، Git status، environment و dependencies را بررسی کن.
- مدل، train loop، evaluation، preprocessing، checkpoint loading و logها را پیدا کن.
- تفاوت کد با `ABLATION_TRAINING_PROTOCOL.md` را گزارش کن.
- فایل Word را inventory کن: sections، tables، figures، placeholders، `XX` و `TBD`.
- موجودبودن datasetها، pretrained weights و checkpointها را بررسی کن.
- هیچ refactor نامرتبطی انجام نده.

خروجی:

```text
reports/repository_audit.md
reports/architecture_inventory.json
reports/manuscript_inventory.md
reports/manuscript_placeholders.csv
```

نسخه فعلی مقاله حدود 913 پاراگراف، 10 جدول و 5 تصویر دارد؛ این اعداد را با نسخه واقعی verify کن.

## مرحله ۲ — Research Specification

تولید کن:

```text
docs/research_spec.md
configs/training_protocol.yaml
configs/ablation_matrix.yaml
```

موارد قطعی:

- split ثابت train/validation/test
- seedهای `42`, `3407`, `2026`
- input برابر 352×352
- checkpoint selection با میانگین validation Dice روی Kvasir و ClinicDB
- عدم استفاده از test برای انتخاب checkpoint یا threshold
- protocol یکسان برای تمام variantها
- مقاله Word منبع اصلی متن است؛ هر تغییر در `manuscript/word_update_queue.md` ثبت شود.

## پروتکل آموزش پیشنهادی

```text
Input: 352×352
Encoder: ImageNet-pretrained
Max epochs: 150
Encoder freeze: 10 epochs
Encoder LR: 1e-5
Decoder LR: 1e-4
Weight decay: 1e-4
Optimizer: AdamW
Early stopping: 30 epochs
Seeds: 42, 3407, 2026
TTA during ablation: disabled
```

## معماری و آزمایش‌ها

Baseline باید ConvNeXt-Tiny، lightweight U-Net decoder و normal skip باشد و MSC، BSEI، Detail Branch، GDF و DS نداشته باشد.

ترتیب incremental:

```text
Baseline → +MSC → +BSEI → +Detail Branch → +GDF → +Deep Supervision
```

کنترل‌ها:

- Backbone: ResNet34، EfficientNet و ConvNeXt-Tiny
- Skip: normal، attention gate و proposed BSEI
- GDF: addition، concatenation، attention fusion و proposed GDF
- MSC branch count: 2، 3، 4 و 5
- MSC dilation sets: `(1,2,3)`, `(1,3,5)`, `(1,3,7)`
- detail channels: بدون detail، 16، 32 و 64
- deep supervision: صفر، یک، دو و سه auxiliary head

## حلقه خودکار و retry

State machine:

```text
AUDIT → PLAN → IMPLEMENT → TEST → PILOT → EXPERIMENT → VALIDATE → AGGREGATE → WORD_UPDATE → REVIEW → DONE
```

در خطا:

```text
TEST/VALIDATE/WORD_UPDATE failure
→ DIAGNOSE → PATCH → targeted retest → regression test
```

حداکثر retry برابر 3 است. اگر یک root cause دو بار تکرار شد، workflow باید `BLOCKED` شود و traceback، command، مسیر فایل و پیشنهاد اصلاح را در `MONITORING.md` و `.agentic/state.json` ثبت کند. نتیجه جعلی نساز و بی‌نهایت retry نکن.

## Gateها

### Gate 1 — Architecture

- همه variantها instantiate می‌شوند.
- tensor shape، output و checkpoint load درست است.
- Params و FLOPs محاسبه می‌شود.

### Gate 2 — Smoke test

- forward، backward و یک epoch کوتاه موفق است.
- loss/gradient finite است.
- checkpoint save/load موفق است.

### Gate 3 — Pilot

- همه variantها ابتدا با seed 42 اجرا می‌شوند.
- نتیجه pilot قبل از full run بررسی می‌شود.

### Gate 4 — Full experiments

- هر variant با هر سه seed اجرا شده است.
- raw log، checkpoint و metadata حفظ شده‌اند.

### Gate 5 — Statistics

- میانگین هر seed محاسبه شده است.
- mean±std بین seedها محاسبه شده است.
- confidence interval/effect size در صورت نیاز ثبت شده است.

### Gate 6 — Word update

- متن نهایی، محل درج، جدول، شکل، caption و evidence manifest آماده است.
- قبل از replace مقاله، backup ساخته و DOCX render شده است.

### Gate 7 — Reviewer QA

- متن، جدول و شکل سازگارند.
- هیچ `XX`، `TBD` یا claim بدون evidence باقی نمانده است.
- reference، DOI و citation audit انجام شده است.

## قرارداد Word-ready

هر مرحله باید طبق `templates/word_update_contract.md` این artifactها را تولید کند:

```text
manuscript/sections/<stage>.md
manuscript/tables/<stage>.csv
manuscript/tables/<stage>.tex
manuscript/figures/<stage>.*
manuscript/captions/<stage>.md
manuscript/word_update_queue.md
```

اگر داده کافی نیست، عدد نساز؛ placeholder دقیق با owner و evidence موردنیاز ثبت کن.

## خروجی نهایی

- کد و configهای reproducible
- raw و aggregated results
- جدول‌های CSV و LaTeX
- شکل معماری، ablation، size analysis و qualitative analysis
- متن نهایی Word
- گزارش reviewer و reproducibility
- ثبت license و citation کدهای GitHub
