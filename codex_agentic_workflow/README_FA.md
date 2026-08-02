# ConvNeXt UGBR Multi-Agent Workflow V3

این بسته برای قرارگرفتن در ریشه پروژه `ConvNeXt_Unet` طراحی شده است.

## هدف

1. توقف امن campaign قدیمی بدون حذف نتایج و checkpointها.
2. audit کامل استفاده صحیح از ConvNeXt در ورودی 352×352.
3. اجرای pilot چهار مدل:
   - `baseline`
   - `baseline_ugbr`
   - `baseline_best_existing`
   - `baseline_best_existing_ugbr`
4. استفاده از loss زیر برای مدل‌های دارای UGBR:

```text
L = Lseg(final) + 0.4 Lseg(initial) + 0.2 Lboundary + 0.1 Lconsistency
```

5. ادامه سه-seed فقط در صورت عبور pilot از gate تعریف‌شده.

## استفاده

پوشه `codex_agentic_workflow_v3` را در ریشه پروژه کپی کنید، سپس فایل زیر را به Codex بدهید:

```text
codex_agentic_workflow_v3/AGENTIC_WORKFLOW.md
```

Codex باید از `MASTER_PROMPT.md` به‌عنوان prompt اجرایی استفاده کند.

## فقط دو فایل برای کاربر

- `RUN_STATUS.md`: وضعیت کوتاه و جاری، حداکثر 15 خط.
- `CAMPAIGN_CONSOLE.log`: خروجی کامل ترمینال visible.

بقیه فایل‌ها داخلی هستند و کاربر لازم نیست آن‌ها را بررسی کند.
