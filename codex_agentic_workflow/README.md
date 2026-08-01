# Agentic Workflow for ConvNeXt-UNet Polyp Segmentation

این بسته برای استفاده در Codex آماده شده است. هدف آن مدیریت یک پروژه پژوهشی کامل از audit کد و تکمیل پیاده‌سازی تا اجرای ablation، تولید جدول و شکل، و تکمیل مقاله است.

## شروع سریع

1. این پوشه را در ریشه repository با نام `codex_agentic_workflow/` کپی کنید.
2. فایل `AGENTIC_WORKFLOW.md` را به Codex بدهید و صریحاً بگویید آن را به‌عنوان دستورالعمل پروژه اجرا کند.
3. قبل از تغییر کد، مقدار LRSE/BSEI، مسیر دیتاست‌ها، وزن pretrained و checkpointها را مشخص کنید.
4. Codex باید ابتدا `reports/repository_audit.md`، سپس `docs/research_spec.md` و بعد plan اجرایی را تولید کند.

## قوانین غیرقابل‌مذاکره

- نتیجه، Dice یا عدد مقاله هرگز جعل، دست‌کاری یا برای صعودی‌کردن مصنوعی انتخاب نمی‌شود.
- همه ablationها با protocol، split، loss، augmentation و seedهای یکسان اجرا می‌شوند.
- هر عدد مقاله باید از فایل نتیجه‌ی تأییدشده استخراج شود.
- کد GitHub فقط پس از بررسی license، commit و سازگاری وارد پروژه می‌شود.
- تا عبور از Gate هر مرحله، مرحله بعد شروع نمی‌شود.

فایل‌های اصلی:

- `AGENTIC_WORKFLOW.md`: دستورالعمل کامل برای Codex
- `configs/ablation_matrix.yaml`: ماتریس آزمایش‌ها
- `templates/experiment_result_schema.json`: schema نتایج
- `templates/github_dependency_record.md`: ثبت کدهای خارجی
- `templates/manuscript_claim_rules.md`: قوانین تولید متن مقاله
