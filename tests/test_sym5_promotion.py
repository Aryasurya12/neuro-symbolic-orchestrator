"""SYM-5: Model promotion and rollback tests."""

import pytest
import json
import os
import shutil

from src.symbolic.optihive.telemetry_training import (
    promote_candidate_model, rollback_to_previous,
    CHECKPOINT_PATH, CANDIDATE_PATH, BACKUP_PATH,
    MIN_PROMOTION_ACCURACY, SYM5_MODEL_VERSION
)
from src.symbolic.telemetry.feedback import CLASS_LABELS


def _write_valid_candidate(path: str, val_accuracy: float = 0.95):
    candidate = {
        "model_version": SYM5_MODEL_VERSION,
        "feature_ordering": [
            "resource_scale", "service_scale", "budget_tightness", "provider_count",
            "has_latency_constraint", "has_sla_constraint",
            "constraint_density", "is_highly_constrained",
        ],
        "class_labels": CLASS_LABELS,
        "W": [[0.0] * 3] * 8,
        "b": [0.1, 0.2, 0.7],
        "mean": [0.0] * 8,
        "scale": [1.0] * 8,
        "metadata": {
            "val_accuracy": val_accuracy,
            "training_samples": 50,
        },
    }
    with open(path, "w") as f:
        json.dump(candidate, f)


def test_candidate_rejection_low_accuracy(tmp_path):
    candidate = str(tmp_path / "candidate.json")
    checkpoint = str(tmp_path / "checkpoint.json")
    backup = str(tmp_path / "backup.json")
    _write_valid_candidate(candidate, val_accuracy=0.50)  # below threshold

    result = promote_candidate_model(
        min_accuracy=MIN_PROMOTION_ACCURACY,
        candidate_path=candidate,
        checkpoint_path=checkpoint,
        backup_path=backup,
    )
    assert result["promoted"] is False
    assert "val_accuracy" in result["reason"].lower() or "threshold" in result["reason"].lower()


def test_candidate_promoted_when_valid(tmp_path):
    candidate = str(tmp_path / "candidate.json")
    checkpoint = str(tmp_path / "checkpoint.json")
    backup = str(tmp_path / "backup.json")
    _write_valid_candidate(candidate, val_accuracy=0.95)

    result = promote_candidate_model(
        candidate_path=candidate,
        checkpoint_path=checkpoint,
        backup_path=backup,
    )
    assert result["promoted"] is True
    assert os.path.exists(checkpoint)
    with open(checkpoint) as f:
        data = json.load(f)
    assert data["model_version"] == SYM5_MODEL_VERSION


def test_backup_created_on_promotion(tmp_path):
    candidate = str(tmp_path / "candidate.json")
    checkpoint = str(tmp_path / "checkpoint.json")
    backup = str(tmp_path / "backup.json")

    # Create an "existing" checkpoint
    _write_valid_candidate(checkpoint, val_accuracy=0.80)
    _write_valid_candidate(candidate, val_accuracy=0.95)

    promote_candidate_model(
        candidate_path=candidate,
        checkpoint_path=checkpoint,
        backup_path=backup,
    )
    assert os.path.exists(backup)


def test_no_candidate_returns_not_promoted(tmp_path):
    result = promote_candidate_model(
        candidate_path=str(tmp_path / "nonexistent.json"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        backup_path=str(tmp_path / "backup.json"),
    )
    assert result["promoted"] is False


def test_corrupt_candidate_returns_not_promoted(tmp_path):
    candidate = str(tmp_path / "bad.json")
    with open(candidate, "w") as f:
        f.write("{not_valid json}")
    result = promote_candidate_model(
        candidate_path=candidate,
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        backup_path=str(tmp_path / "backup.json"),
    )
    assert result["promoted"] is False


def test_rollback_restores_previous(tmp_path):
    checkpoint = str(tmp_path / "checkpoint.json")
    backup = str(tmp_path / "backup.json")

    _write_valid_candidate(backup, val_accuracy=0.80)
    # Current checkpoint has a different version
    _write_valid_candidate(checkpoint, val_accuracy=0.50)

    result = rollback_to_previous(checkpoint_path=checkpoint, backup_path=backup)
    assert result["rolled_back"] is True
    with open(checkpoint) as f:
        data = json.load(f)
    assert data["metadata"]["val_accuracy"] == 0.80


def test_rollback_fails_without_backup(tmp_path):
    result = rollback_to_previous(
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        backup_path=str(tmp_path / "does_not_exist.json"),
    )
    assert result["rolled_back"] is False


def test_missing_schema_field_prevents_promotion(tmp_path):
    candidate = str(tmp_path / "bad_schema.json")
    with open(candidate, "w") as f:
        json.dump({"model_version": "sym5-v1", "W": []}, f)  # missing required fields
    result = promote_candidate_model(
        candidate_path=candidate,
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        backup_path=str(tmp_path / "backup.json"),
    )
    assert result["promoted"] is False
    assert "missing" in result["reason"].lower()
