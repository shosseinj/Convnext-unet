# External model status

| Model | Repository | Official weights found | Official prediction maps found | Inference successful | Notes |
|---|---|---:|---:|---:|---|
| CTNet | https://github.com/Fhujinwu/CTNet | No released checkpoint link in README | Yes | Not required | Repository-shipped `result_map.zip`; verified Kvasir (100) and CVC-ClinicDB (62) maps. |
| MEGANet | https://github.com/UARK-AICV/MEGANet | Yes | Yes | Not run | Official Res2Net map archive used for Kvasir. Archive has no CVC-ClinicDB directory; checkpoint downloaded, but its backbone download failed when disk reached 0 bytes free. |
| PraNet-V2 | https://github.com/ai4colonoscopy/PraNet-V2 | Yes | README states yes, but no direct map link exposed | No | Official `RES-V2.pth` downloaded. Git clone stalled twice and source ZIP was truncated; no prediction was fabricated. |
| MLB-Net | https://github.com/cloneiq/MLBNet | No model checkpoint link in README | Yes | Not required / maps unavailable | Official Google Drive folder returned HTTP 401 to `gdown`; no third-party substitute used. |
| CIFFormer | https://github.com/lonlin404/CIFFormer | No | No | No | Official repository contains test code but supplies no trained checkpoint path or downloadable maps. Training was not attempted. |

`N/A` panels in the generated figure mean that no verified official prediction was available for that image/model combination.
