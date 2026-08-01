# گزارش پیشرفت پروژه ConvNeXt-UNet

تاریخ گزارش: ۱۰ مرداد ۱۴۰۵ / 2026-08-01  
وضعیت کلی: stage جاری `EXPERIMENT` است؛ اجرای رسمی `baseline / seed 42` تکمیل و عمیقاً اعتبارسنجی شده، اجرای `baseline / seed 3407` در آخرین snapshot به epoch 78 رسیده، شمار رسمی `1/18` است و Gate آزمایش هنوز PASS نشده است.

## خلاصه مدیریتی

repository بررسی شد و مشخص شد کد فعلی از نظر تنظیمات آموزش، انتخاب checkpoint،
seedها و جداسازی validation/test در نسخه اولیه با پروتکل پژوهشی یکسان نبود.
نام رسمی ماژول در workflow و artifactهای جدید فقط BSEI است و SimpleFusion
به‌عنوان normal skip/fusion در نظر گرفته شد.

یک architecture factory ساخته شد که شش variant اصلی ablation را بدون ویرایش
دستی کد می‌سازد. هر شش معماری در اندازه واقعی 352 در 352 تست شدند و Gate 1 و
Gate 2 را گذراندند. بااین‌حال این نتایج فقط صحت فنی معماری را ثابت می‌کنند و
هنوز نشان‌دهنده Dice، IoU یا کیفیت نهایی مدل‌ها نیستند.

## کارهای انجام‌شده

### ۱. Repository audit

- ساختار پروژه، Git، مدل‌ها، train loop، evaluation و checkpoint loading بررسی شد.
- تعداد image/maskهای datasetها کنترل شد:
  - Kvasir-SEG: تعداد 1000 جفت
  - CVC-ClinicDB: تعداد 612 جفت
  - CVC-300: تعداد 60 جفت
  - CVC-ColonDB: تعداد 380 جفت
  - ETIS-LARIBPOLYPDB: تعداد 196 جفت
- وزن pretrained مربوط به ConvNeXt-Tiny موجود است.
- 229 checkpoint و 46 log متنی در repository پیدا شد.
- preprocessing فعلی شناسایی شد: BGR به RGB، resize به 352 در 352، مقیاس
  صفر تا یک و ImageNet normalization داخل مدل.

### ۲. Research specification و configها

فایل‌های زیر ساخته شدند:

- `reports/repository_audit.md`
- `reports/architecture_inventory.json`
- `docs/research_spec.md`
- `configs/training_protocol.yaml`
- `configs/ablation_matrix.yaml`

پروتکل اصلی ثبت‌شده شامل موارد زیر است:

- seedهای 42، 3407 و 2026
- حداکثر 150 epoch
- freeze شدن encoder برای 10 epoch
- encoder LR برابر 1e-5
- decoder LR برابر 1e-4
- weight decay برابر 1e-4
- AdamW
- early stopping برابر 30 epoch
- خاموش بودن TTA در ablation
- انتخاب checkpoint با میانگین مساوی validation Dice روی Kvasir و ClinicDB
- ممنوعیت استفاده از external test datasets برای انتخاب threshold یا checkpoint

### ۳. Architecture factory

فایل `models/architecture_factory.py` اضافه شد. شش معماری زیر اکنون از config
ساخته می‌شوند:

1. Baseline
2. Baseline + MSC
3. Baseline + MSC + BSEI
4. Baseline + MSC + BSEI + Detail Branch
5. Baseline + MSC + BSEI + Detail Branch + GDF
6. Full model + سه deep-supervision head

رفتار پیش‌فرض `ConvNeXtUNet` همان Baseline + MSC باقی مانده است تا checkpointهای
فعلی تا جای ممکن سازگار بمانند.

### ۴. Gate 1: صحت معماری

وضعیت: PASS

| Variant | Parameters | MACs | خروجی |
|---|---:|---:|---|
| Baseline | 29,190,098 | 13.60B | یک mask |
| Baseline + MSC | 29,572,851 | 13.65B | یک mask |
| + BSEI | 29,579,763 | 13.68B | یک mask |
| + Detail | 29,604,819 | 14.45B | یک mask |
| + GDF | 29,601,107 | 14.34B | یک mask |
| Full | 29,601,782 | 14.34B | یک main و سه auxiliary mask |

تمام خروجی‌ها در ورودی واقعی 352 در 352 دارای ابعاد صحیح بودند. strict
checkpoint round-trip نیز موفق بود و checkpoint ناقص خطای واضح تولید کرد.

شواهد: `reports/architecture_gate.json`

### ۵. Gate 2: Smoke test

وضعیت: PASS

برای هر شش variant این موارد اجرا شد:

- دو batch مصنوعی
- forward و backward
- finite بودن loss و gradient
- optimizer step با AdamW
- ذخیره و بارگذاری مجدد checkpoint

شواهد: `reports/smoke_gate.json`

تمام 9 unit test موجود نیز پاس شدند.

## نتایج فعلی چه چیزی را ثابت می‌کنند؟

نتایج فعلی ثابت می‌کنند که:

- variantها واقعاً قابل ساخت و از هم متمایز هستند.
- BSEI، DetailBranch، GDF و deep supervision وارد graph محاسباتی می‌شوند.
- شکل خروجی برای segmentation صحیح است.
- backward و optimizer step بدون crash انجام می‌شوند.
- checkpoint یک معماری با strict loading قابل بازیابی است.
- محاسبه تعداد پارامتر و MACs قابل تکرار است.

نتایج فعلی ثابت نمی‌کنند که:

- Full model از Baseline بهتر است.
- Dice یا IoU مقاله بهبود یافته است.
- training پایدار و بدون overfitting است.
- سه seed نتیجه مشابه می‌دهند.
- checkpointهای قدیمی با variantهای جدید قابل مقایسه منصفانه هستند.

## ایرادها و ریسک‌های فعلی

### ۱. مسیر اصلی آموزش هنوز ناسازگار است

`main_torch.py` هنوز به runner جدید وصل نشده و defaultهای آن با پروتکل اختلاف دارد:

- epochs برابر 50000 به‌جای 150
- freeze/warmup برابر 24 به‌جای 10
- weight decay برابر 5e-4 و refine برابر 1e-3 به‌جای 1e-4
- early stopping برابر 40 به‌جای 30
- TTA به‌صورت پیش‌فرض فعال
- training به‌صورت پیش‌فرض خاموش و load روشن

تا اصلاح این مسیر نباید pilot رسمی اجرا شود.

### ۲. checkpoint selection اشتباه است

کد فعلی بهترین checkpoint را با IoU روی validation ترکیبی انتخاب می‌کند. پروتکل
نیاز دارد ابتدا Dice هر dataset جدا محاسبه و سپس این مقدار استفاده شود:

`(kvasir_val_dice + clinicdb_val_dice) / 2`

### ۳. reproducibility کامل نیست

- split manifest ذخیره نمی‌شود.
- Python، NumPy، PyTorch، CUDA و DataLoader یک seed manager واحد ندارند.
- permutationهایی بدون seed صریح وجود دارند.
- hash مربوط به split و config داخل checkpoint ذخیره نمی‌شود.

### ۴. خطر test leakage

external datasetها از CLI اصلی قابل انتخاب هستند. guard سختی وجود ندارد که مانع
انتخاب threshold یا checkpoint روی CVC-300، ColonDB یا ETIS شود.

### ۵. loss بالای Full در smoke test

loss اولیه Full model در داده مصنوعی از variantهای تک‌خروجی بیشتر بود. علت اصلی
این است که loss چهار خروجی با هم جمع شده است و داده نیز تصادفی است. این مورد
شکست محسوب نمی‌شود چون loss و gradient finite بودند، ولی runner واقعی باید
وزن‌های deep supervision ثابت و مستند، مثلاً `(1.0, 0.1, 0.05, 0.02)`، اعمال کند.

### ۶. نتایج قدیمی برای مقاله قابل استفاده مستقیم نیستند

checkpointها و logهای قبلی الزاماً با split، seed و protocol جدید ساخته نشده‌اند.
تا زمانی که metadata و raw results آن‌ها قابل اثبات نباشد نباید در جدول اصلی
ablation با نتایج جدید مخلوط شوند.

## کارهای باقی‌مانده

### مرحله بعدی ضروری: reproducible training runner

1. ساخت split manifest ثابت برای هر سه seed.
2. افزودن seed manager کامل.
3. ساخت DataLoaderهای جدا برای Kvasir validation و ClinicDB validation.
4. اتصال architecture factory به training runner.
5. اعمال freeze ده epoch و LR groupهای ثابت.
6. اعمال loss و deep-supervision weights یکسان.
7. محاسبه selection score صحیح.
8. ذخیره resolved config، Git commit، split hash و metricها در checkpoint.
9. افزودن guard برای external test datasets.
10. اجرای یک epoch واقعی کوتاه روی subset کوچک برای تکمیل Gate 2 داده‌محور.

### Gate 3: Pilot

پس از تکمیل runner، تمام شش variant ابتدا فقط با seed 42 اجرا می‌شوند. نتایج pilot
برای کشف crash، overfitting، مصرف حافظه و رفتار loss هستند و نباید به‌عنوان نتیجه
نهایی مقاله استفاده شوند.

### Gate 4: Full experiments

بعد از تأیید pilot، هر variant با هر سه seed اجرا می‌شود. برای هر اجرا config،
metadata، log، best checkpoint و raw per-image metrics ذخیره خواهد شد.

### Gateهای 5 و 6

- محاسبه mean و sample standard deviation بین seedها
- تحلیل اندازه پولیپ در سه گروه small، medium و large
- تولید CSV، JSON، LaTeX، table و figure
- تطبیق اعداد مقاله با artifactهای تأییدشده
- Reviewer QA و گزارش reproducibility

## پیشنهاد اقدام فعلی

گام بعدی نباید اجرای مستقیم training فعلی باشد. ابتدا باید runner جدید ساخته و
یک epoch واقعی کوتاه با seed 42 اجرا شود. پس از عبور این تست، اجرای pilot شش
variant منطقی و قابل اعتماد خواهد بود.

## وضعیت مرحله

```text
Stage: Audit + Research Design + Architecture Gate + Synthetic Smoke Gate
Status: PASS
Files changed: audit/spec/config/factory/validators/tests
Evidence: reports/architecture_gate.json, reports/smoke_gate.json
Risks: old training path, checkpoint selection, incomplete seeding, test leakage
Next gate: reproducible training runner and data-backed smoke epoch
```

## به‌روزرسانی مرحله reproducibility

پروتکل آموزش اکنون loss، augmentation، scheduler، batch size و وزن‌های deep
supervision را نیز به‌صورت صریح ثبت می‌کند. ماژول seed manager و تولید split
manifest اضافه شده است. manifestها بر اساس نام فایل ساخته می‌شوند، train و
validation overlap ندارند و برای تشخیص تغییرات دارای SHA-256 هستند.

فایل‌های این مرحله:

- `research_pipeline/reproducibility.py`
- `tools/create_split_manifests.py`
- `tests/test_reproducibility.py`

وضعیت: PASS

تمام 11 unit test پاس شدند و manifestهای واقعی زیر تولید شدند:

| Seed | Kvasir train/val | ClinicDB train/val | SHA-256 |
|---:|---:|---:|---|
| 42 | 900/100 | 550/62 | `ae39a534009f8a46c6aa7b87b9c78a886b251bdff24ad639024ba623715b5bbe` |
| 3407 | 900/100 | 550/62 | `d846380aff8b4f50313e5c5847e5cfa3bbdd45ddf565e8bd0c587541871dc4a7` |
| 2026 | 900/100 | 550/62 | `7eedc1f59930b29a60d102c1e160b897774f5b240d21a19c2e58915d9cd61819` |

فایل‌ها در `configs/splits/` ذخیره شده‌اند. تست‌ها تکرارپذیری یک seed، تفاوت
seedهای مختلف و نبود overlap بین train و validation را تأیید کردند.

```text
Stage: Reproducible Splits
Status: PASS
Commands run: py_compile, unittest discover, create_split_manifests.py
Evidence: configs/splits/development_seed_*.json
Next gate: config-driven training runner
```

## به‌روزرسانی training runner و smoke واقعی

فایل `train_research.py` و ماژول‌های `research_pipeline/data.py` و
`research_pipeline/losses.py` ساخته شدند. runner جدید موارد زیر را enforce می‌کند:

- استفاده از architecture factory و variant ID معتبر
- استفاده اجباری از وزن pretrained encoder
- خواندن split فقط از manifest مربوط به seed
- DataLoader جدا برای validation مربوط به Kvasir و ClinicDB
- selection score برابر میانگین مساوی Dice دو dataset
- AdamW با encoder LR برابر 1e-5 و decoder LR برابر 1e-4
- freeze شدن encoder طبق schedule
- deep-supervision weights برابر `(1.0, 0.1, 0.05, 0.02)`
- ذخیره resolved config، Git commit و manifest hash کنار checkpoint
- عدم وجود مسیر external test dataset در training runner
- محدود شدن override تعداد epoch به حالت `--smoke`

یک smoke واقعی یک‌epochی و محدود روی داده‌های واقعی برای هر شش variant اجرا شد.
هر اجرا شامل یک training batch و یک validation batch از هر dataset بود:

| Variant | Status | Train loss | Selection Dice محدود |
|---|---|---:|---:|
| Baseline | PASS | 2.8057 | 0.2335 |
| Baseline + MSC | PASS | 1.2332 | 0.0130 |
| + BSEI | PASS | 0.8949 | 0.2607 |
| + Detail | PASS | 6.1126 | 0.2298 |
| + GDF | PASS | 0.9665 | 0.0136 |
| Full | PASS | 1.4785 | 0.1231 |

این Diceها نتیجه پژوهشی نیستند؛ مدل‌ها فقط یک optimizer step دیده‌اند و validation
فقط شامل 8 تصویر از هر dataset بوده است. کاربرد این اعداد فقط اثبات اجرای کامل
data-to-checkpoint است.

در smoke واقعی، OpenCV برای تعدادی TIFF مربوط به ClinicDB هشدار ناسازگاری
`ExtraSamples` نمایش داد. فایل‌ها قابل خواندن بودند و batch کامل شد، ولی پیش از
full experiment باید یک audit تصویری/کانالی روی تمام TIFFهای ClinicDB انجام شود.

```text
Stage: Data-backed Smoke
Status: PASS
Evidence: results/raw/<variant>/seed_42/smoke/{resolved_config,history,summary,best.pth}
Risks: TIFF ExtraSamples warnings; smoke metrics are not scientific results
Next gate: dataset TIFF audit, then seed-42 Pilot
```

## به‌روزرسانی audit فرمت ClinicDB

audit کامل فایل‌ها انجام شد و مشکل دقیق مشخص شد:

- Kvasir-SEG: هر 1000 فایل تصویر JPEG معتبر هستند؛ همه pairها خوانده می‌شوند.
- CVC-ClinicDB: هر 612 فایل با پسوند `.png` ذخیره شده‌اند، اما magic header آن‌ها
  `II 2A 00` و فرمت واقعی آن‌ها TIFF little-endian است.
- همه 612 image و mask توسط OpenCV قابل decode هستند.
- همه ClinicDB image/maskها ابعاد 384 در 288 دارند و mismatch ابعادی وجود ندارد.
- داده decodeشده ClinicDB تک‌کاناله با بازه 0 تا 255 است؛ مسیر training آن را به
  RGB سه‌کاناله تبدیل می‌کند.
- Pillow این فایل‌های mislabeled را باز نمی‌کند و OpenCV هشدار TIFF metadata می‌دهد.

اصل داده‌ها تغییر یا تبدیل داده نشدند. این وضعیت در
`reports/dataset_file_audit.json` ثبت شده و loader رسمی همچنان decode صریح OpenCV
را استفاده می‌کند. بنابراین از نظر readability مانع Pilot نیست، اما provenance
داده باید در گزارش نهایی ذکر شود.

```text
Stage: Dataset File Audit
Status: PASS_WITH_WARNING
Evidence: reports/dataset_file_audit.json
Risk: 612 TIFF files mislabeled with .png extension and malformed ExtraSamples metadata
Decision: preserve originals, decode with audited OpenCV path, record provenance warning
Next gate: hardware check and seed-42 Pilot
```

## تعریف Pilot

RTX 4090 با 24GB حافظه توسط runtime شناسایی شد. Pilot به‌صورت مستقل از official
run تعریف شد: seed برابر 42، سه epoch، حداکثر 25 training batch در هر epoch و
validation کامل Kvasir و ClinicDB. نتایج Pilot مجاز به ورود مستقیم به مقاله نیستند.

## review نسخه اول Pilot

هر شش variant سه epoch اجرا شدند و هیچ crash، OOM، non-finite loss یا خطای
checkpoint رخ نداد. بهترین selection Dice باینری محدود به‌ترتیب زیر بود:

| Variant | Best selection Dice |
|---|---:|
| Baseline | 0.0008 |
| Baseline + MSC | 0.0423 |
| + BSEI | 0.0410 |
| + Detail | 0.1062 |
| + GDF | 0.0049 |
| Full | 0.0718 |

این Pilot برای عبور Gate 3 کافی نیست. تمام سه epoch در بازه freeze ده‌epochی
قرار داشتند و unfreeze encoder و scheduler بررسی نشدند. علاوه بر آن، افت Dice
باینری با وجود کاهش train loss می‌تواند ناشی از collapse در threshold ثابت 0.5
باشد. Pilot به 12 epoch اصلاح شد و soft Dice و predicted-positive fraction به
evaluation افزوده شدند.

```text
Stage: Pilot v1 Review
Status: NEEDS_REVIEW
Decision: do not start official/full experiments
Fix: 12-epoch Pilot crossing encoder unfreeze; add soft Dice and positive fraction
```

Pilot v2 برای Baseline تا epoch 12 اجرا شد. unfreeze در epoch 11 و scheduler بدون
خطا عمل کردند، اما predicted-positive fraction نزدیک صفر باقی ماند؛ soft Dice در
epoch 11 برابر 0.2164 برای Kvasir و 0.1621 برای ClinicDB بود، درحالی‌که Dice
باینری selection فقط 0.0048 شد.

در review مشخص شد boundary term در runner جدید فقط weighted BCE را پیاده کرده
بود، درحالی‌که کد audited پروژه weighted BCE به‌علاوه weighted IoU دارد. این
اختلاف قبل از ادامه اصلاح شد. نتایج Pilotهای قبل از این اصلاح برای تصمیم معماری
نامعتبر هستند و باید بازتولید شوند.

پس از افزودن weighted IoU، تمام 13 unit test پاس شدند و Baseline Pilot v2 دوباره
اجرا شد. در epoch 11 که encoder برای اولین بار unfreeze شد:

- selection Dice: `0.0788`
- Kvasir Dice: `0.1263`
- ClinicDB Dice: `0.0313`
- Kvasir soft Dice: `0.2414`
- ClinicDB soft Dice: `0.1812`
- predicted-positive fraction در Kvasir به `0.0382` افزایش یافت.

این شواهد نشان می‌دهد مسیر unfreeze و foreground prediction فعال است. نوسان
epoch 12 نشان می‌دهد Pilot محدود 25-batch برای رتبه‌بندی نهایی مناسب نیست، اما
برای بررسی سلامت training path قابل استفاده است. پنج variant دیگر باید با همین
نسخه loss بازتولید شوند.

## نتیجه Gate 3: Pilot v2

هر شش variant با loss اصلاح‌شده، seed 42، دوازده epoch، validation کامل و عبور
از encoder unfreeze اجرا شدند. aggregateها در
`results/aggregated/pilot_seed42.csv` و `pilot_seed42.json` ذخیره شدند.

| Variant | Best epoch | Selection Dice | Kvasir Dice | ClinicDB Dice |
|---|---:|---:|---:|---:|
| Baseline | 11 | 0.0788 | 0.1263 | 0.0313 |
| Baseline + MSC | 12 | 0.5033 | 0.5547 | 0.4520 |
| + BSEI | 3 | 0.0592 | 0.0774 | 0.0411 |
| + Detail | 11 | 0.2821 | 0.3493 | 0.2149 |
| + GDF | 7 | 0.3218 | 0.3961 | 0.2474 |
| Full | 11 | 0.3124 | 0.3747 | 0.2502 |

نتیجه review:

- تمام variantها forward/backward، checkpoint، freeze و unfreeze را گذراندند.
- هیچ OOM، crash یا non-finite loss رخ نداد.
- Baseline+MSC در Pilot محدود سریع‌تر همگرا شد.
- BSEI به‌تنهایی در این budget محدود ضعیف و ناپایدار بود.
- Detail، GDF و Full بعد از unfreeze foreground prediction معنادار ایجاد کردند.
- این رتبه‌بندی علمی نیست چون هر epoch فقط 25 training batch داشت.
- تمام ردیف‌ها `eligible_for_manuscript=false` هستند.

```text
Stage: Pilot v2
Status: PASS
Evidence: results/raw/*/seed_42/pilot and results/aggregated/pilot_seed42.*
Decision: architecture/training path is eligible for official experiments
Warning: Pilot ranking must not be copied into manuscript tables
Next gate: 6 variants x 3 seeds official training
```

## هزینه مرحله official

مرحله بعد شامل 18 اجرای مستقل تا حداکثر 150 epoch است. با توجه به زمان Pilot،
هر اجرای کامل احتمالاً حدود 60 تا 90 دقیقه روی RTX 4090 زمان می‌برد؛ کل صف حدود
18 تا 27 ساعت GPU time نیاز دارد و ممکن است early stopping آن را کاهش دهد.
اجرای official باید با monitor، resume، ثبت log مستقل و جلوگیری از اجرای هم‌زمان
دو job روی یک GPU انجام شود.

برای این مرحله، `train_research.py` به checkpoint قابل-resume در پایان هر epoch
مجهز شد. `tools/run_official_queue.py` نیز 18 اجرا را به‌شکل sequential روی یک
GPU اجرا می‌کند، run کامل را skip می‌کند، run ناقص را resume می‌کند و رخدادهای
صف را در `results/raw/official_queue.jsonl` می‌نویسد.

## شروع صف رسمی

صف official در تاریخ 2026-08-01 ساعت 08:40 UTC شروع شد.

- Queue process wrapper PID: `37416`
- Queue Python PID: `22780`
- نخستین job: `baseline`, seed `42`
- Training wrapper PID: `28280`
- Training Python/GPU PID: `34804`
- stdout: `results/raw/official_queue_stdout.log`
- stderr: `results/raw/official_queue_stderr.log`
- queue events: `results/raw/official_queue.jsonl`

در زمان ثبت گزارش، processها فعال و baseline seed 42 روی GPU در حال اجرا بود.
صف فقط یک job را در هر لحظه اجرا می‌کند. در صورت قطع process، اجرای مجدد همان
دستور با `--resume` از `last.pth` ادامه می‌دهد.

```text
Stage: Official Experiments
Status: IN_PROGRESS
Completed official runs: 0/18 at queue start
Current run: baseline seed 42
Next update: after completed run summaries are available
```

## ادامه workflow از آخرین وضعیت — 2026-08-01 09:10 UTC

داشبورد و state از روی evidence واقعی بازسازی شدند. مراحل `AUDIT`، `PLAN`،
`IMPLEMENT`، `TEST` و `PILOT` دارای artifact قابل‌بررسی هستند و stage جاری
`EXPERIMENT` است. صف رسمی همچنان فعال است؛ در آخرین snapshot، اجرای baseline با
seed 42 به epoch 40 رسیده بود و هنوز هیچ run رسمی کامل برای ورود به مقاله وجود
نداشت.

- هشت تست مجاز architecture/reproducibility/loss با discovery هدفمند پاس شدند.
- تست `tests/test_video_evaluation.py` طبق ممنوعیت Video QA Tester اجرا نشد.
- مقاله واقعی در Downloads پیدا شد و بدون تغییر، مستقیماً از OOXML audit شد:
  329 پاراگراف بدنه، 10 جدول، 5 تصویر و 21 پاراگراف دارای placeholder.
- render بسته‌بندی‌شده به‌دلیل نبود dependencyهای سند در runtime پروژه اجرا نشد؛
  بنابراین visual PASS برای Word ثبت نشده است.
- `reports/manuscript_inventory.md`، `reports/manuscript_placeholders.csv`،
  `.agentic/state.json` و صف Word-ready ساخته شدند.
- هیچ مقدار Pilot وارد مقاله نشد و فایل Word نیز تغییر نکرد.
- هیچ commit یا push انجام نشد.

```text
Stage: EXPERIMENT
Status: IN_PROGRESS
Previous state: PILOT PASS (state file previously absent)
Tasks checked: audit/protocol/implementation/tests/pilot/official queue/Word inventory
Tasks completed: state reconstruction, manuscript inventory, 8 approved tests
Files changed: monitoring, state, reports, research spec, naming test/config, manuscript queue artifacts
Commands executed: git/process/log inspection, OOXML audit, py_compile, targeted unittest discovery
Tests passed: 8/8 approved unit tests; prior architecture/smoke/pilot gates verified
Evidence paths: reports/*, results/raw/*, results/aggregated/pilot_seed42.*, .agentic/state.json
Word sections/tables/figures updated: queue only; source DOCX unchanged
Remaining placeholders: 21 manuscript paragraphs plus user-owned author/declaration fields
Failure IDs: F-001, F-002, F-003
Next stage: EXPERIMENT remains current until all required official runs complete
Reason for transition: official queue is active; no validated official aggregate exists yet
```

اعتبارسنجی پایانی همین نوبت: JSON و YAML معتبر، SVG قابل parse، هشت تست مجاز
دوباره PASS، و صف رسمی فعال بود. baseline seed 42 در آخرین snapshot به epoch 43
رسید. این مقدار فقط وضعیت اجرای training است و نتیجه نهایی/مقاله‌ای محسوب نمی‌شود.

## پایش و اصلاح evidence — 2026-08-01 09:26 UTC

Gate مرحله EXPERIMENT بررسی شد و به‌دلیل کامل‌نشدن 18 اجرای رسمی هنوز PASS نیست.
اجرای فعال baseline با seed 42 در آخرین snapshot به epoch 64 رسیده و process آن
فعال است. checkpoint انتخابی فقط از validation Dice دو dataset توسعه استفاده
می‌کند و test set در انتخاب دخالت ندارد.

در Diagnose مشخص شد metadata اولیه، environment versions، تعداد پارامترها، روش
complexity و fingerprint فایل‌های منبع را کامل ذخیره نمی‌کند. fingerprint دقیق
run فعال پیش از patch در `active_run_source_fingerprint.json` ثبت شد و تولید
metadata برای runهای بعدی کامل شد. Retest اول ناسازگاری separator مسیر ویندوز را
نشان داد؛ با POSIX normalization اصلاح شد. Retest دوم و هر هشت regression test
مجاز PASS شدند. Video QA Tester اجرا نشد.

artifactهای Word-ready پروتکل شامل متن، CSV، LaTeX، SVG و caption ساخته و در
Word Update Queue ثبت شدند. هیچ نتیجه عملکردی و هیچ مقدار موقت وارد مقاله نشد.

Snapshot نهایی این نوبت: epoch 66، process فعال، summary رسمی 0 از 18. بنابراین
stage بدون تغییر `EXPERIMENT / IN_PROGRESS` باقی ماند و انتقال خودکار انجام نشد.

## orchestration تخصصی و recovery — 2026-08-01 09:38 UTC

سه عامل تخصصی به‌صورت read-only سلامت run، قرارداد evidence و الزامات مقاله را
ممیزی کردند. اجرای baseline seed 42 سالم، GPU-bound و بدون traceback/OOM/NaN بود؛
checkpointها قابل load و آرشیوهای آن‌ها سالم بودند. در snapshot نهایی این بخش،
run به epoch 80 رسیده و هنوز 0 از 18 summary رسمی کامل است.

دو شکاف recovery اصلاح شدند: checkpointهای آینده همه RNGها و generator آموزش را
ذخیره/restore می‌کنند و resume در صورت تغییر identity یا fingerprint رد می‌شود؛
queue آینده نیز completion ناقص را skip نمی‌کند، lock دارد و eventهای command،
device و validation را ثبت می‌کند. تست Windows lock ابتدا دو بار child output را
بست؛ علت `os.kill(pid, 0)` بود و با `OpenProcess` اصلاح شد. در پایان 2 تست queue،
2 تست recovery و 8 regression test مجاز PASS شدند. run فعال تغییر یا restart نشد.

ممیزی Word نشان داد نسخه اصلی در 24 پاراگراف نام قدیمی دارد و هنوز هیچ BSEI در
آن نیست. محل‌ها در WQ-001 گسترش یافتند، اما جایگزینی تا code-to-equation review و
مرحله Word Update انجام نمی‌شود. هیچ عدد عملکردی وارد مقاله نشد و DOCX تغییر نکرد.

## آماده‌سازی evaluation و aggregation — ادامه session در 2026-08-01

در زمان ادامه run فعال، evaluator رسمی پنج dataset ساخته شد. این ابزار فقط پس از
وجود summary کامل و معتبر اجرا می‌شود، threshold ثابت 0.5 و TTA غیرفعال دارد و
برای هر تصویر Dice، IoU، precision، recall، specificity، pixel accuracy، MAE و
size bin را ذخیره می‌کند. dry-run روی run ناقص عمداً رد شد. aggregator نیز فقط
سه seed کامل را می‌پذیرد و mean، sample standard deviation و 95% t interval را
از داده رسمی محاسبه می‌کند؛ incomplete aggregate عمداً رد شد.

دو matrix ناسازگار به یک source of truth تبدیل شدند و queue اکنون 18 job
incremental را از `configs/ablation_matrix.yaml` می‌خواند. کنترل‌های special هنوز
implementation کامل ندارند و انجام‌شده علامت نخورده‌اند.

artifactهای complexity شامل متن، CSV، LaTeX، SVG و caption مستقیماً از
`reports/architecture_gate.json` تولید و مقداربه‌مقدار verify شدند. این اعداد فقط
هزینه معماری هستند و claim عملکردی نیستند. در آخرین snapshot، run رسمی فعال و در
epoch 86 بود؛ summary رسمی همچنان 0 از 18 است. تعداد تست‌های مجاز ثبت‌شده 18 و
تعداد failure نهایی صفر است.

## تطبیق مستقیم BSEI با مقاله — ادامه session در 2026-08-01

implementation واقعی BSEI خط‌به‌خط و با tensor introspection بررسی شد. response
تک‌کاناله نیست: برای هر کانال خروجی یک depthwise response تولید می‌شود. چهار سطح
768→384، 384→192، 192→96 و 96→96 با ورودی مصنوعی اجرا شدند؛ shape، تعداد group
و padding همگی با جدول آماده مقاله تطبیق داشتند. متن، equations، CSV، LaTeX، SVG
و caption در artifactهای BSEI و WQ-001 ثبت شدند. نام رسمی در همه artifactهای جدید
فقط BSEI است. DOCX همچنان تغییر نکرده است.

آخرین snapshot run: baseline seed 42، epoch 89، process فعال، 0/18 summary کامل.

## کنترل‌های ویژه و بازآزمایی — ادامه session در 2026-08-01

اجرای رسمی `baseline / seed 42` بدون وقفه ادامه دارد و در آخرین بررسی به epoch 105 رسیده است؛ هر چهار process مربوط به queue و training فعال‌اند و هنوز هیچ summary رسمی کامل نشده است (0/18). مقادیر validation فعلی صرفاً برای پایش سلامت run هستند و وارد مقاله نشده‌اند.

ماتریس کنترل‌های ویژه تکمیل و بررسی شد. حالت `control_gdf_concatenation` از نظر مسیر محاسباتی با `baseline_msc_bsei_detail` یکسان بود و برای جلوگیری از اجرای تکراری به `control_reuse` منتقل شد؛ در نتیجه 14 کنترل مستقل باقی ماند. gate معماری smoke برای هر 14 کنترل روی CPU و ورودی 64×64 PASS شد و shape خروجی، strict checkpoint round-trip، خطای checkpoint ناسازگار، تعداد پارامتر و MAC ثبت شد. این gate فقط evidence معماری است و هیچ نتیجه عملکردی تولید نمی‌کند.

وزن‌های رسمی محلی ResNet-34 و EfficientNet-B0 با SHA-256 و strict-load در `pretrained/manifest.json` ثبت شده‌اند. سازگاری عقب‌رو نیز با strict-load کامل checkpointهای pilot برای `baseline` و `full` تأیید شد؛ هیچ کلید missing یا unexpected وجود نداشت.

پس از اصلاح شمارش قدیمی کنترل‌ها (F-012)، هر 21 تست مجاز PASS شدند. `Video QA Tester` اجرا نشده است. فایل DOCX همچنان بدون تغییر است و همه اعداد عملکردی، عدم‌قطعیت‌ها، جدول‌های نتیجه و claimهای مقایسه‌ای تا پایان آزمایش‌های رسمی placeholder یا blocked باقی می‌مانند.

queue جداگانه کنترل‌ها نیز آماده شد، اما یک gate سخت دارد و تا زمانی که هر 18 اجرای incremental واقعاً PASS نشده باشند launch نمی‌شود. این queue فقط 14 کنترل مستقل را برای سه seed، یعنی 42 job یکتا، زمان‌بندی می‌کند و reuseها را دوباره train نمی‌کند. خطای import در تست اولیه با شناسه F-013 ثبت و اصلاح شد؛ پس از retest، مجموع تست‌های مجاز 23/23 PASS است. آخرین وضعیت ثبت‌شده اجرای فعال epoch 108 با چهار process سالم و 0/18 اجرای کامل است.

artifactهای Word-ready تعریف کنترل‌ها نیز شامل متن Methods، جدول CSV، جدول LaTeX، شکل SVG و caption تولید و با matrix canonical تطبیق داده شدند. این artifactها هیچ عدد عملکردی ندارند و صرفاً طراحی آزمایش را مستند می‌کنند. پایش مستقل اجرای فعال در epoch 109 سلامت چهار process، فعالیت GPU، رشد history و checkpoint و نبود traceback، OOM، CUDA error یا NaN را تأیید کرد؛ completion artifact هنوز وجود ندارد.

orchestrator نهایی‌سازی incremental نیز آماده شد تا فقط پس از تکمیل 18/18، برای هر run ابتدا deep validation و سپس evaluation ثابت را اجرا و در پایان aggregate سه-seed را بسازد. اجرای واقعی `--check-only` در وضعیت فعلی به‌درستی با پیام نبود `summary.json` متوقف شد و هیچ نتیجه ناقصی promote نشد. دو تست جدید همراه regression کامل PASS شدند و شمار تست‌های مجاز به 25/25 رسید. اجرای فعال در آخرین snapshot به epoch 113 رسیده و summary هنوز ساخته نشده است.

آخرین snapshot این نوبت: `baseline / seed 42` در epoch 115، هر چهار process فعال، summary رسمی هنوز غایب و شمار تکمیل 0/18 است؛ بنابراین stage همچنان EXPERIMENT / IN_PROGRESS باقی ماند.

gate معماری کنترل‌ها سپس در اندازه رسمی 352×352 روی CPU اجرا شد و هر 14 کنترل مستقل PASS شدند. از همین evidence جدول CSV و LaTeX پیچیدگی، شکل SVG و caption تولید شد. خطای اولیه escape در f-string با F-014 ثبت و با کوچک‌ترین تغییر اصلاح شد؛ compile، شمار 14 ردیف، تطبیق MACها با gate و parse شکل همگی PASS شدند. این اعداد فقط هزینه معماری هستند و claim عملکردی محسوب نمی‌شوند. اجرای رسمی هم‌زمان به epoch 117 رسید و هنوز 0/18 است.

قرارداد aggregation تقویت شد: خروجی رسمی علاوه بر mean، sample standard deviation و CI، اکنون status، روش آماری، timestamp و مسیر و SHA-256 هر 15 summary ورودی را ثبت می‌کند. تست هدفمند hash provenance و regression کامل PASS شدند و شمار تست‌های مجاز 26/26 شد. run فعال در آخرین بررسی epoch 121 بود و summary رسمی هنوز وجود نداشت.

مولد artifact نتایج incremental آماده شد تا پس از وجود شش aggregate معتبر، جدول CSV و LaTeX، شکل روند، تحلیل size-stratified، caption و متن Results را مستقیماً از evidence بسازد. اجرای زنده در وضعیت فعلی با پیام صریح BLOCKED به‌علت نبود aggregate متوقف شد و هیچ فایل عددی ناقص تولید نکرد. تست guard و regression کامل شمار مجاز را به 27/27 PASS رساند. آخرین epoch مشاهده‌شده 124 و شمار رسمی همچنان 0/18 است.

مسیر موفق تولید artifact نیز با fixture کامل شش variant و پنج dataset آزمایش شد؛ 30 ردیف نتیجه، 30 ردیف size-stratified و SVG معتبر تولید شدند. شمار تست‌های یکتای مجاز اکنون 28 است. همچنین scan تمام فایل‌های متنی repository نشان داد هیچ نام legacy ماژول باقی نمانده و نام رسمی فقط BSEI است؛ جایگزینی داخل DOCX طبق gate Word هنوز انجام نشده است. اجرای فعال به epoch 126 رسیده و summary ندارد.

finalizer جداگانه کنترل‌ها اضافه شد. این ابزار تنها پس از وجود 42 run معتبر، برای هر run deep validation و evaluation و سپس aggregate سه-seed را اجرا می‌کند. compile و تست هدفمند PASS شدند و اجرای زنده `--check-only` در وضعیت فعلی به‌درستی با 0/42 متوقف شد. شمار تست‌های یکتای مجاز 29 است؛ run فعال در epoch 127 و همچنان 0/18 است.

## تکمیل نخستین اجرای رسمی — 2026-08-01

اجرای `baseline / seed 42` بدون وقفه تا 150 epoch کامل شد و summary رسمی با وضعیت PASS و completion=true تولید کرد. deep validator فرمول انتخاب checkpoint، شمار نمونه‌های validation، تمام 150 ردیف history، تطبیق بهترین epoch، best/last checkpoint، هویت run و SHA-256 artifactها را بررسی و PASS کرد. recovery contract این run به‌درستی `legacy-uninterrupted` ثبت شد و به‌عنوان run قابل resume معرفی نشد.

صف با return code صفر به `baseline / seed 3407` منتقل شد. resolved config اجرای جدید environment، model metadata، source fingerprints و pretrained-weight evidence را دارد و در آخرین snapshot به epoch 1 رسیده است. پس از completion نخست، تمام 29 تست مجاز دوباره PASS شدند و Video QA Tester اجرا نشد. وضعیت رسمی اکنون 1/18 است؛ نتایج عملکردی هنوز تا تکمیل سه seed و aggregation وارد مقاله نمی‌شوند.

برای تداوم خودکار workflow یک campaign guardian با PID 35652 و PID lock فعال شد. این process هر run کامل را deep-validate و در ledger ثبت می‌کند، ولی تا پایان 18/18 هیچ evaluation یا job کنترل روی GPU آغاز نمی‌کند. سپس به‌ترتیب finalization incremental، 42 اجرای کنترل و finalization کنترل‌ها را اجرا خواهد کرد. تست هدفمند و regression کامل 30/30 PASS هستند. run جاری `baseline / seed 3407` در آخرین snapshot epoch 4 بود و guardian نیز 1/18 را ثبت کرده است.

بسته Word-ready محدودیت‌ها و مرز evidence ساخته شد. متن به‌صراحت image-level بودن split، نبود شناسه patient/procedure/video، محدودیت سه seed، ثابت بودن threshold، نبود TTA، حذف S-measure پیاده‌سازی‌نشده و حذف latency/FPS/peak-memory بدون benchmark را بیان می‌کند. جدول تصمیم، LaTeX، SVG و caption نیز اعتبارسنجی شدند و WQ-007 ثبت شد. وضعیت placeholderهای مرتبط به READY_FOR_WORD، OMIT_NO_EVIDENCE یا QUALIFIED_LIMITATION تغییر کرد؛ هیچ عددی جعل نشد.

gate مستقل artifactهای غیرعددی manuscript هر هفت بسته audit، BSEI، loss، experiment، complexity، controls و limitations را از نظر وجود فایل، CSV، SVG، placeholder و نام legacy بررسی و PASS کرد. numerical results عمداً BLOCKED_BY_EXPERIMENT ماند. شمار تست‌های یکتا 31 و اجرای فعال baseline seed 3407 در epoch 8 است.

اجرای baseline seed 3407 به epoch 10 رسید و metadata توسعه‌یافته، hashهای source و weights، پارامترها، GPU/environment، history و checkpointهای ZIP سالم تأیید شدند. epoch 10 مطابق protocol آخرین epoch encoder-frozen است؛ run سالم و شمار رسمی 1/18 باقی مانده است.

با workflow اسناد، پاراگراف‌های هدف DOCX مستقیماً و read-only از OOXML استخراج و با SHA-256 منبع ثبت شدند. پس از تکمیل front matter، نقشه شامل 24 هدف است: 12 مورد آماده درج Word، هفت مورد وابسته به نتایج/evaluation و پنج مورد نیازمند اطلاعات نویسندگان/کاربر. F-015 خطای نحوی نخستین دستور استخراج بود و بدون هیچ تغییر DOCX اصلاح شد. فایل اصلی مقاله همچنان دست‌نخورده است.

پس از مشاهده صفحه اول، نقشه هدف با عنوان، Abstract، نام نویسندگان، affiliations و corresponding email کامل‌تر شد. PDF 29 صفحه‌ای از DOCX منبع در حالت read-only با Word تولید شد و صفحه اول با ابعاد 1224×1584 به PNG تبدیل و بصری بررسی شد؛ clipping یا overlap دیده نشد، اما placeholderها و نام legacy مطابق انتظار در منبع دست‌نخورده حاضرند. این فقط PARTIAL PASS برای مسیر render است و جای full visual QA نسخه نهایی را نمی‌گیرد.

اتصال browser با F-016 و دو تلاش automation تمام‌صفحه با F-018 ثبت شدند؛ در هر دو loop کامل Office خروجی تولید نشد و فقط processهای renderer پس از تطبیق PID و عنوان بسته شدند. Word موجود کاربر حفظ شد. اجرای رسمی در تمام این بررسی‌ها ادامه یافت و اکنون baseline seed 3407 در epoch 26، guardian فعال و وضعیت 1/18 است.

فهرست جداگانه اطلاعات غیرقابل‌استنتاج نویسندگان ساخته شد: نام و ترتیب نویسندگان، affiliations، corresponding author/email، URL و شرایط انتشار کد، و نقش‌های CRediT. این پنج هدف هرگز از repository حدس زده نمی‌شوند؛ آزمایش‌ها را متوقف نمی‌کنند اما برای DOCX نهایی بدون placeholder ضروری‌اند. آخرین epoch ثبت‌شده run فعال 27 است.
## به‌روزرسانی وضعیت زنده — 2026-08-01

اجرای رسمی `baseline / seed 42` با deep validation واقعی PASS است و شمارش رسمی همچنان `1/18` است؛ فایل smoke در شمارش اجرای رسمی وارد نشد. اجرای فعال `baseline / seed 3407` بدون توقف در epoch 36 مشاهده شد و guardian جدید با PID 28016 فعال است. هیچ evaluation هم‌زمان با صف GPU آغاز نشده است.

کل regression مجاز دوباره اجرا شد: 35/35 تست PASS؛ فایل `tests/test_video_evaluation.py` اجرا نشد. Gate بسته‌های مقاله برای هفت بسته غیرعددی PASS است و بخش نتایج عددی تا تکمیل آزمایش‌ها صریحاً `BLOCKED_BY_EXPERIMENT` باقی ماند. ابزارهای تولید شکل qualitative، artifactهای نتایج incremental و control و audit نهایی آماده‌اند، اما هیچ عدد یا claim ناقص وارد مقاله نشده است.

خطای F-019 ناشی از انتظار قدیمی 120 ردیف در تست control بود؛ ماتریس canonical شامل 25 پیکربندی نمایشی در پنج dataset است و 125 ردیف صحیح است. assertion اصلاح و کل regression مجاز مجدداً PASS شد. کارهای باقی‌مانده: تکمیل 18 اجرای incremental، ارزیابی و aggregation، اجرای 42 کنترل مستقل، تولید artifactهای عددی، دریافت اطلاعات نویسندگان، backup و ویرایش DOCX، render کامل و بازبینی بصری همه صفحات.
## رفع مسیر render کامل Word — 2026-08-01

مسیر نهایی render بدون نصب dependency جدید تثبیت شد. Word نسخه منبع را به PDF read-only صادر می‌کند و ابزار `tools/render_pdf_winrt.ps1` با API داخلی `Windows.Data.Pdf` هر صفحه PDF را مستقیماً به PNG با ابعاد 1224×1584 تبدیل می‌کند. هر 29 صفحه منبع بصری بررسی شد: جدول‌ها، شکل‌ها، معادلات، page breakها، فونت‌ها و حاشیه‌ها بدون clipping یا overlap هستند. F-001 و F-018 برای مسیر render رفع شدند، اما نسخه نهایی ویرایش‌شده DOCX همچنان باید پس از backup و درج evidence دوباره کامل render و صفحه‌به‌صفحه بررسی شود.

در طول این کار صف GPU متوقف نشد. اجرای `baseline / seed 3407` به epoch 47 رسید، guardian با PID 28016 فعال است و شمار رسمی همچنان 1/18 است. هیچ عدد حین آموزش وارد مقاله نشد و Video QA Tester اجرا نشد.
## سلامت checkpoint در epoch 50 — 2026-08-01

اجرای `baseline / seed 3407` در epoch 50 به‌صورت read-only عمیق بررسی شد. همه metricهای history متناهی بودند؛ `best.pth` و `last.pth` هم از نظر ساختار ZIP و هم با `torch.load(..., map_location='cpu')` سالم بارگذاری شدند. `last.pth` شامل RNG state، scheduler state و cumulative elapsed time است. queue، training و guardian با PIDهای مورد انتظار فعال‌اند و هیچ Traceback، OOM، NaN یا campaign error مشاهده نشد. آموزش در پایان ثبت به epoch 51 رسیده و شمار رسمی 1/18 باقی مانده است. مقادیر metric این snapshot فقط progress evidence هستند و برای مقاله یا انتخاب علمی استفاده نمی‌شوند.
## اصلاح قطعی نام BSEI — 2026-08-01

در audit پیش از Word مشخص شد چند artifact برای `BSEI` یک expanded form استنباطی نوشته‌اند که با قانون ثابت «نام رسمی فقط BSEI» سازگار نبود. تمام expanded formها از protocol، متن و caption مقاله، Word Update Queue، عنوان هدف DOCX و docstringهای مرتبط حذف شدند. scan نهایی در repository برای هر دو expanded form صفر hit و scan manuscript برای نام legacy صفر hit داشت. JSON هدف Word parse شد، Manuscript Artifact Gate PASS ماند و تست هدفمند آن نیز PASS شد. F-020 ثبت و رفع شد. آموزش هم‌زمان بدون توقف به epoch 54 رسید.
## Gate جلوگیری از ویرایش زودهنگام Word — 2026-08-01

ابزار `tools/check_word_update_readiness.py` اضافه شد تا مرحله Word به‌صورت fail-closed اجرا شود. Gate قبل از هر تغییر، SHA-256 منبع DOCX، status تمام targetها، PASS بودن artifactهای عددی و qualitative و کامل بودن پنج گروه اطلاعات نویسندگان را بررسی می‌کند. سه تست مستقل برای مسیر READY، blockerهای آزمایش/نویسنده و hash mismatch PASS شدند. اجرای زنده با hash صحیح منبع، به‌درستی چهار blocker `UNRESOLVED_DOCX_TARGETS`، `NUMERICAL_RESULTS_NOT_READY`، `QUALITATIVE_PACKAGE_NOT_READY` و `AUTHOR_INPUTS_MISSING` را ثبت کرد و هیچ DOCX را تغییر نداد.

پس از اصلاح wrapper شمارش که FutureWarning غیرکشنده `timm` را خطا تلقی کرده بود (F-021)، کل regression مجاز شامل 38/38 تست PASS شد. Video QA Tester اجرا نشد. آموزش هم‌زمان به epoch 57 رسید و شمار رسمی 1/18 باقی ماند.
## قرارداد backup و اطلاعات نویسندگان — 2026-08-01

فایل `manuscript/author_inputs.template.json` برای دریافت ساختاریافته نام‌ها، affiliations، corresponding author، code availability و نقش‌های CRediT ساخته شد. ابزار `tools/prepare_word_update_backup.py` فقط بعد از `READY` شدن Gate اجازه ساخت backup timestamped می‌دهد، هرگز فایل موجود را overwrite نمی‌کند و برابری SHA-256 منبع و backup را در manifest ثبت می‌کند. اجرای زنده فعلی به‌درستی با status مسدود متوقف شد و حتی directory backup نیز نساخت.

یک حالت fail-open در اعتبارسنجی containerهای توخالی template شناسایی و با F-022 ثبت شد؛ اکنون validation بازگشتی است و رشته‌ها، mappingها و آرایه نقش‌های خالی را رد می‌کند. چهار تست readiness و سه تست backup PASS شدند و regression کامل مجاز به 42/42 PASS رسید. Video QA Tester اجرا نشد. اجرای GPU بدون توقف به epoch 61 رسید.

در snapshot زنده بعدی، history اجرای `baseline / seed 3407` شامل 65 epoch کامل و metricهای ثبت‌شده بود. چهار process مربوط به train و official queue و guardian با PID 28016 فعال ماندند؛ `summary.json` و `validation.json` برای run جاری هنوز ساخته نشده‌اند، شمار deep-validation رسمی `1/18` است، و جست‌وجوی logهای اخیر هیچ Traceback، CUDA OOM، NaN یا Inf نشان نداد. checkpointهای `best.pth` و `last.pth` از نظر ZIP سالم‌اند و `last.pth` هم‌زمان با epoch 65 تازه شده است. این بررسی فقط evidence سلامت اجرای درحال‌آموزش است و هیچ مقدار موقت به مقاله منتقل نشد.

Gate پیش از ویرایش Word سخت‌تر شد: اکنون targetهای پاراگرافی تکراری یا خارج از محدوده، action ناشناخته، replacement بدون payload، داشتن هم‌زمان دو payload، و source گمشده یا خالی را fail-closed رد می‌کند. شش تست هدفمند و کل regression مجاز شامل 44/44 تست PASS شدند؛ `test_video_evaluation.py` اجرا نشد. اجرای زنده Gate همچنان فقط به‌علت نتایج آزمایشی/qualitative و اطلاعات نویسندگان BLOCKED است و هیچ backup یا تغییر DOCX ایجاد نشد. در پایان تست‌های CPU-only، run رسمی به epoch 68 رسیده، guardian فعال و fatal scan پاک بود.

مرحله Word اکنون یک plan materializer مستقل و fail-closed دارد. برای replacementهای چندپاراگرافی حالت صریح `markdown_body` و برای جدولهای نیازمند درج ساختاری حالت `source_reference` ثبت شد. `tools/build_word_update_plan.py` فقط پس از READY شدن کامل Gate، heading فایل Markdown را حذف، مرز پاراگراف‌ها را حفظ و SHA-256 هر payload و target map را ثبت می‌کند. اجرای زنده آن با exit code 2 متوقف شد و `reports/word_update_plan.json` ایجاد نشد. 9 تست هدفمند مربوط به readiness/plan و regression کامل مجاز 47/47 PASS شدند؛ Video QA اجرا نشد. اجرای رسمی هم‌زمان به epoch 72 رسید و guardian فعال باقی ماند.

ممیزی ساختار واقعی DOCX نشان داد target پاراگراف 50 caption جدول است و خود جدول یک عنصر جداگانه 15x4 پس از آن است؛ همچنین map پاراگرافی به‌تنهایی حذف همه نام‌های قدیمی را تضمین نمی‌کرد. این نقص با F-023 ثبت و اصلاح شد: caption جداگانه، patch سلولی با مقدار قبلی موردانتظار، و جایگزینی document-wide نام رسمی فقط با `BSEI` تعریف شد. applicator قطعی OOXML نیز اضافه شد که تنها روی خروجی جدید کار می‌کند، hash منبع و backup manifest واقعی را اجباری می‌سنجد، drift جدول و placeholder/نام قدیمی باقی‌مانده را رد می‌کند و ZIP خروجی را بررسی می‌کند. سه تست applicator و کل regression مجاز 51/51 PASS شد. اجرای زنده طبق انتظار قبل از ساخت خروجی BLOCKED ماند؛ source DOCX و backup directory تغییر نکردند. run رسمی در پایان به epoch 78 رسید، guardian فعال و fatal scan پاک بود.
