from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "53_fafem_residual_frequency_guided_mscb_stage3_pranet_split_3seeds.ps1"


def test_exp53_campaign_keeps_tqdm_stderr_out_of_tee_log_pipeline():
    source = RUNNER.read_text(encoding="utf-8")
    assert "2>&1 |" not in source
    assert "Tee-Object -FilePath $campaignLog -Append" in source
