# Agentic Workflow — ConvNeXt-UNet Polyp Segmentation

## نقش Codex

تو مدیر یک workflow پژوهشی reproducible هستی. پروژه `Convnext-unet-main` را از روی کد، داده و نتایج واقعی جلو ببر. نقش‌های منطقی زیر را به‌ترتیب اجرا کن:

1. Research Planner
2. Repository Auditor
3. Implementation Engineer
4. Experiment Engineer
5. Evaluation and Statistics Engineer
6. Figure and Table Engineer
7. Manuscript Engineer
8. Reviewer QA Engineer

در محیطی که agentهای موازی در دسترس نیستند، این نقش‌ها را به‌صورت sequential اجرا کن و برای هر نقش artifact مشخص بساز.

## اصل اول: audit قبل از implementation

قبل از تغییر کد:

- ساختار repository و Git status را بررسی کن.
- مدل، train loop، evaluation، preprocessing، checkpoint loading و logها را پیدا کن.
- تفاوت کد فعلی با `ABLATION_TRAINING_PROTOCOL.md` را گزارش کن.
- مشخص کن `LRSE` همان `BSEI` است یا ماژول دیگری است؛ اگر مبهم بود، توقف کن و سؤال بپرس.
- مسیر و موجودبودن datasetها، pretrained weights و checkpointها را بررسی کن.
- هیچ refactor نامرتبطی انجام نده.

خروجی اجباری:

```text
reports/repository_audit.md
reports/architecture_inventory.json
```

## اصل دوم: طراحی پژوهش

بعد از audit، این فایل‌ها را تولید کن:

```text
docs/research_spec.md
configs/training_protocol.yaml
configs/ablation_matrix.yaml
```

در design باید این موارد قطعی باشند:

- split ثابت train/validation/test
- seedهای `42`, `3407`, `2026`
- input size برابر 352×352
- checkpoint selection با میانگین validation Dice روی Kvasir و ClinicDB
- عدم استفاده از test datasets برای انتخاب checkpoint یا threshold
- پروتکل یکسان برای تمام variantها

## پروتکل آموزش پیشنهادی

مگر اینکه کاربر یا داده‌ی پروژه مقدار دیگری را تأیید کند:

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

## معماری variantها

Baseline باید ConvNeXt-Tiny به‌همراه lightweight U-Net decoder و normal skip باشد و MSC، LRSE/BSEI، Detail Branch، GDF و DS نداشته باشد.

ترتیب incremental:

```text
Baseline
Baseline + MSC
Baseline + MSC + LRSE/BSEI
Baseline + MSC + LRSE/BSEI + Detail Branch
Baseline + MSC + LRSE/BSEI + Detail Branch + GDF
Full model + Deep Supervision
```

Ablationهای جداگانه:

- Backbone: ResNet34، EfficientNet و ConvNeXt-Tiny
- Skip: normal، attention gate و proposed LRSE/BSEI
- GDF: addition، concatenation، attention fusion و proposed GDF
- MSC branch count: 2، 3، 4 و 5 شاخه
- MSC dilation sets: `(1,2,3)`, `(1,3,5)`, `(1,3,7)`
- Detail channels: بدون detail، 16، 32 و 64
- Deep supervision: صفر، یک، دو و سه auxiliary head

حالت proposed و حالت‌های تکراری را دوباره اجرا نکن؛ در ماتریس با یک شناسه مشترک reuse کن.

## Gateهای اجباری

### Gate 1 — Architecture

- همه variantها instantiate می‌شوند.
- ابعاد output درست است.
- checkpoint load با خطای واضح انجام می‌شود.
- تعداد پارامتر و FLOPs محاسبه می‌شود.

### Gate 2 — Smoke test

- حداقل یک forward و backward موفق است.
- loss finite است.
- checkpoint save/load موفق است.
- یک epoch کوتاه بدون crash اجرا می‌شود.

### Gate 3 — Pilot

- تمام variantها ابتدا با seed 42 اجرا می‌شوند.
- هیچ نتیجه‌ای برای مقاله قبل از بررسی pilot استفاده نمی‌شود.

### Gate 4 — Full experiments

- هر variant با هر سه seed اجرا شده است.
- metadata، log، checkpoint و نتیجه خام حفظ شده‌اند.
- validation و test از هم جدا هستند.

### Gate 5 — Statistics

- میانگین هر seed ابتدا محاسبه می‌شود.
- سپس mean و sample standard deviation بین seedها محاسبه می‌شود.
- جدول‌ها به شکل `mean ± std` تولید می‌شوند.

### Gate 6 — Manuscript

- تمام عددهای متن از JSON/CSV تأییدشده می‌آیند.
- هیچ `[TBD]`، `XX%` یا ادعای بدون evidence باقی نمی‌ماند.
- متن با جدول‌ها و شکل‌ها تطابق دارد.

## استفاده از GitHub

برای هر کد خارجی این مراحل را انجام بده:

1. repository و commit/release دقیق را ثبت کن.
2. license را بررسی کن.
3. سازگاری نسخه Python/PyTorch/CUDA را بررسی کن.
4. کد را در صورت امکان به‌عنوان dependency نگه دار؛ copy کردن را محدود کن.
5. test یا benchmark قبل و بعد از integration اجرا کن.
6. citation و license را در `templates/github_dependency_record.md` ثبت کن.

کد خارجی نباید بدون تأیید کاربر جایگزین بخش اصلی معماری شود.

## نتایج و مقاله

ساختار خروجی را حفظ کن:

```text
results/raw/
results/aggregated/
results/tables/
results/figures/
manuscript/sections/
manuscript/tables/
manuscript/figures/
reports/
```

تحلیل اندازه پولیپ را با نسبت area mask به area تصویر انجام بده:

```text
Small: < 5%
Medium: 5% تا 20%
Large: > 20%
```

برای هر گروه تعداد نمونه، Dice mean±std و IoU mean±std گزارش کن.

## قرارداد گزارش پیشرفت

در پایان هر مرحله، Codex باید این موارد را بنویسد:

```text
Stage:
Status: PASS | BLOCKED | NEEDS_REVIEW
Files changed:
Commands run:
Evidence:
Risks:
Next gate:
```

در صورت BLOCKED شدن، حدس نزن و نتیجه بساز؛ blocker را با مسیر فایل و خطای دقیق گزارش کن.

## قرارداد تحویل نهایی

تحویل نهایی باید شامل این موارد باشد:

- کد قابل اجرا
- configهای تمام آزمایش‌ها
- log و metadata
- checkpointهای منتخب
- raw و aggregated results
- CSV و LaTeX tableها
- شکل‌های مقاله
- متن اصلاح‌شده مقاله
- گزارش reproducibility
- گزارش license و citation کدهای GitHub
- دستورهای دقیق اجرای مجدد
