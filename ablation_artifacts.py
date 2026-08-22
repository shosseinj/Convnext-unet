"""Validation and atomic persistence for ablation evaluation artifacts."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


DATASETS = (
    "Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB",
    "ETIS-LaribPolypDB",
)
METRICS = ("mDice", "mIoU", "F_beta_w", "S_alpha", "mE_phi", "maxE_phi", "MAE")
COMPLEXITY_FIELDS = (
    "trainable_parameters", "total_parameters", "macs", "flops", "gmacs", "gflops",
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_evaluation_summary(path, experiment_name, seed, fingerprint=None):
    path = Path(path)
    if not path.is_file():
        return ValidationResult(False, "evaluation_summary.json is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ValidationResult(False, f"evaluation_summary.json is invalid: {exc}")
    if payload.get("experiment_name") != experiment_name or payload.get("seed") != seed:
        return ValidationResult(False, "evaluation experiment/seed metadata does not match")
    if fingerprint is not None and payload.get("checkpoint_fingerprint") != fingerprint:
        return ValidationResult(False, "evaluation checkpoint fingerprint does not match")
    for field in COMPLEXITY_FIELDS:
        value = payload.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return ValidationResult(False, f"evaluation field is missing or invalid: {field}")
    results = payload.get("results")
    if not isinstance(results, dict):
        return ValidationResult(False, "evaluation results are missing")
    for dataset in DATASETS:
        metrics = results.get(dataset)
        if not isinstance(metrics, dict):
            return ValidationResult(False, f"evaluation dataset is missing: {dataset}")
        for metric in METRICS:
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                return ValidationResult(False, f"evaluation metric is missing or invalid: {dataset}/{metric}")
    return ValidationResult(True, "evaluation summary is valid")
