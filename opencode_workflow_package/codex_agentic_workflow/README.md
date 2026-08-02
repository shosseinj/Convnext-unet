# Agentic Workflow V2 — BSEI-ConvNeXt-UNet Article Completion

این پوشه جایگزین نسخه قبلی workflow است. هدف آن مدیریت خودکار و مرحله‌ای تکمیل implementation، اجرای آزمایش‌ها، تولید شکل/جدول و تکمیل فایل Word مقاله `hossein_paper_revised.docx` است.

## شروع سریع

1. این پوشه را در ریشه repository با نام `codex_agentic_workflow/` کپی کنید و نسخه قبلی همین پوشه را جایگزین کنید.
2. فایل `AGENTIC_WORKFLOW.md` را به Codex بدهید و بگویید آن را به‌عنوان دستورالعمل پروژه اجرا کند.
3. فایل مقاله `hossein_paper_revised.docx` را در workspace قرار دهید.
4. فایل `MONITORING.md` را برای مشاهده وضعیت کارها باز کنید؛ Codex باید آن را در پایان هر مرحله به‌روزرسانی کند.

## قوانین اصلی

- نام رسمی ماژول فقط `BSEI` است؛ نام‌های قدیمی باید در audit شناسایی و با BSEI هماهنگ شوند.
- Video QA Tester در این workflow وجود ندارد.
- نتیجه یا عدد مقاله هرگز جعل یا برای صعودی‌کردن مصنوعی Dice انتخاب نمی‌شود.
- هر عدد مقاله باید از evidence تأییدشده استخراج شود.
- هر مرحله باید خروجی Word-ready شامل متن، جدول، شکل، caption و محل درج تولید کند.
- کد GitHub فقط پس از بررسی license، commit و سازگاری وارد پروژه می‌شود.

## فایل‌های اصلی

- `AGENTIC_WORKFLOW.md`: دستورالعمل کامل Codex
- `MONITORING.md`: داشبورد کارهای انجام‌شده، باقیمانده و blockerها
- `configs/monitoring.yaml`: state machine و retry policy
- `configs/ablation_matrix.yaml`: ماتریس آزمایش‌ها
- `templates/word_update_contract.md`: قرارداد خروجی برای Word
- `templates/stage_report.md`: قالب گزارش هر مرحله
