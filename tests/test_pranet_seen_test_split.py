from pathlib import Path

import pytest

from pranet_seen_test_split import build_seen_test_split, discover_paired_filenames


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
MANIFEST = REPO_ROOT / "configs" / "splits" / "development_seed_42.json"


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="local polyp data are unavailable")
def test_existing_manifest_builds_fixed_seen_test_and_internal_validation_split():
    available = {
        "Kvasir-SEG": discover_paired_filenames(DATA_ROOT, "Kvasir-SEG"),
        "CVC-ClinicDB": discover_paired_filenames(DATA_ROOT, "CVC-ClinicDB"),
    }
    split = build_seen_test_split(MANIFEST, available, internal_validation_seed=42)

    assert len(split["Kvasir-SEG"]["pool"]) == 900
    assert len(split["Kvasir-SEG"]["final_seen_test"]) == 100
    assert len(split["CVC-ClinicDB"]["pool"]) == 550
    assert len(split["CVC-ClinicDB"]["final_seen_test"]) == 62
    assert len(split["combined"]["train"]) == 1305
    assert len(split["combined"]["internal_validation"]) == 145

    kvasir_pool = set(split["Kvasir-SEG"]["pool"])
    kvasir_test = set(split["Kvasir-SEG"]["final_seen_test"])
    clinic_pool = set(split["CVC-ClinicDB"]["pool"])
    clinic_test = set(split["CVC-ClinicDB"]["final_seen_test"])
    train = set(split["combined"]["train"])
    validation = set(split["combined"]["internal_validation"])

    assert not (kvasir_pool & kvasir_test)
    assert not (clinic_pool & clinic_test)
    assert not (train & validation)
    assert not (train & {f"Kvasir-SEG/{name}" for name in kvasir_test})
    assert not (validation & {f"Kvasir-SEG/{name}" for name in kvasir_test})
    assert not (train & {f"CVC-ClinicDB/{name}" for name in clinic_test})
    assert not (validation & {f"CVC-ClinicDB/{name}" for name in clinic_test})

    repeat = build_seen_test_split(MANIFEST, available, internal_validation_seed=42)
    assert split == repeat
