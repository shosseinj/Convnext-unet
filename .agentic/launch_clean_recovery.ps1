$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet'
$env:PYTHONUNBUFFERED = '1'
$python = 'C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe'
$queue = 'C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet\tools\run_official_queue.py'
$outputRoot = 'C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet\results\raw_clean_recovery_20260802'
$log = 'C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet\CAMPAIGN_CONSOLE.log'
Write-Host "CLEAN RECOVERY | baseline_msc_bsei_detail_gdf | seed 2026 | epoch 0 | no resume"
& $python -u $queue --python $python --output-root $outputRoot --only-variant baseline_msc_bsei_detail_gdf --only-seed 2026 --no-resume 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
Write-Host "REAL PYTHON EXIT CODE: $code"
exit $code
