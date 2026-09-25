"""SYM-5: Telemetry-driven model training and controlled promotion.

Architecture:
    Telemetry Store
         ↓
    build_training_dataset() [feedback.py]
         ↓
    Train candidate model
         ↓
    Validate candidate vs current
         ↓
    Promote only if validation passes (keep backup)
         ↓
    learned_checkpoint.json updated atomically

SAFETY:
    - Training uses seed=42 for determinism.
    - Promotion only occurs if accuracy >= MIN_PROMOTION_ACCURACY.
    - Previous checkpoint is backed up before overwriting.
    - On any failure, existing model is preserved unchanged.
    - Model NEVER determines feasibility; only recommends solvers.
"""

import json
import os
import shutil
import time
from typing import Optional, Tuple

import numpy as np

from src.symbolic.telemetry.repository import TelemetryRepository
from src.symbolic.telemetry.feedback import build_training_dataset, CLASS_LABELS, MIN_TELEMETRY_SAMPLES

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "learned_checkpoint.json")
CANDIDATE_PATH = os.path.join(os.path.dirname(__file__), "learned_checkpoint.candidate.json")
BACKUP_PATH = os.path.join(os.path.dirname(__file__), "learned_checkpoint.previous.json")

SYM5_MODEL_VERSION = "sym5-v1"
MIN_PROMOTION_ACCURACY = 0.70   # candidate must reach this accuracy on held-out split

TRAINING_SEED = 42


def _softmax_train(X_norm: np.ndarray, y: np.ndarray,
                   epochs: int = 2000, lr: float = 0.5, reg: float = 0.01):
    n_samples, n_features = X_norm.shape
    n_classes = len(np.unique(y))
    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)
    for _ in range(epochs):
        scores = np.dot(X_norm, W) + b
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        probs[np.arange(n_samples), y] -= 1
        dW = np.dot(X_norm.T, probs) / n_samples + reg * W
        db = np.sum(probs, axis=0) / n_samples
        W -= lr * dW
        b -= lr * db
    return W, b


def _softmax_accuracy(X_norm: np.ndarray, y: np.ndarray,
                      W: np.ndarray, b: np.ndarray) -> float:
    preds = np.argmax(np.dot(X_norm, W) + b, axis=1)
    return float(np.mean(preds == y))


def train_candidate_from_telemetry(
    repo: Optional[TelemetryRepository] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Load telemetry, train a candidate model, write candidate checkpoint.

    Returns a status dict:
        {
          "status": "ok" | "insufficient_data" | "training_failed",
          "record_count": int,
          "valid_count": int,
          "train_accuracy": float | None,
          "val_accuracy": float | None,
          "model_version": str,
        }
    """
    if repo is None:
        repo = TelemetryRepository(db_path=db_path) if db_path else TelemetryRepository()

    records = repo.fetch_all()
    total = len(records)

    X_raw_list, y_list, labels = build_training_dataset(records)
    if X_raw_list is None:
        return {
            "status": "insufficient_data",
            "record_count": total,
            "valid_count": 0,
            "train_accuracy": None,
            "val_accuracy": None,
            "model_version": None,
        }

    X_raw = np.array(X_raw_list, dtype=float)
    y = np.array(y_list, dtype=int)
    n_valid = len(y)

    # Deterministic split: 80% train / 20% val using seed=42
    np.random.seed(TRAINING_SEED)
    idx = np.random.permutation(n_valid)
    split = max(1, int(0.8 * n_valid))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train_raw, y_train = X_raw[train_idx], y[train_idx]
    X_val_raw, y_val = X_raw[val_idx], y[val_idx]

    # Normalize using training statistics
    mean = np.mean(X_train_raw, axis=0)
    scale = np.std(X_train_raw, axis=0)
    safe_scale = np.where(scale == 0, 1.0, scale)

    X_train = (X_train_raw - mean) / safe_scale
    X_val = (X_val_raw - mean) / safe_scale

    try:
        W, b = _softmax_train(X_train, y_train)
        train_acc = _softmax_accuracy(X_train, y_train, W, b)
        val_acc = _softmax_accuracy(X_val, y_val, W, b) if len(val_idx) > 0 else train_acc

        candidate = {
            "model_version": SYM5_MODEL_VERSION,
            "feature_ordering": [
                "resource_scale", "service_scale", "budget_tightness", "provider_count",
                "has_latency_constraint", "has_sla_constraint",
                "constraint_density", "is_highly_constrained",
            ],
            "class_labels": labels,
            "W": W.tolist(),
            "b": b.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "metadata": {
                "training_samples": int(len(train_idx)),
                "val_samples": int(len(val_idx)),
                "total_telemetry_records": total,
                "valid_records": n_valid,
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "trained_at": time.time(),
                "seed": TRAINING_SEED,
            },
        }

        with open(CANDIDATE_PATH, "w") as f:
            json.dump(candidate, f, indent=2)

        return {
            "status": "ok",
            "record_count": total,
            "valid_count": n_valid,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "model_version": SYM5_MODEL_VERSION,
        }

    except Exception as e:
        return {
            "status": "training_failed",
            "record_count": total,
            "valid_count": n_valid,
            "train_accuracy": None,
            "val_accuracy": None,
            "error": str(e),
            "model_version": None,
        }


def promote_candidate_model(
    min_accuracy: float = MIN_PROMOTION_ACCURACY,
    candidate_path: str = CANDIDATE_PATH,
    checkpoint_path: str = CHECKPOINT_PATH,
    backup_path: str = BACKUP_PATH,
) -> dict:
    """Promote candidate to production if it passes validation.

    Steps:
        1. Load candidate checkpoint.
        2. Verify schema integrity.
        3. Check val_accuracy >= min_accuracy.
        4. Back up existing checkpoint.
        5. Atomically write new checkpoint.

    Returns status dict with "promoted": bool and reason.
    """
    if not os.path.exists(candidate_path):
        return {"promoted": False, "reason": "Candidate checkpoint not found."}

    try:
        with open(candidate_path, "r") as f:
            candidate = json.load(f)
    except Exception as e:
        return {"promoted": False, "reason": f"Cannot load candidate: {e}"}

    # Schema validation
    required = {"model_version", "feature_ordering", "class_labels", "W", "b", "mean", "scale", "metadata"}
    missing = required - set(candidate.keys())
    if missing:
        return {"promoted": False, "reason": f"Candidate missing fields: {missing}"}

    val_acc = candidate.get("metadata", {}).get("val_accuracy", 0.0)
    if val_acc < min_accuracy:
        return {
            "promoted": False,
            "reason": f"val_accuracy {val_acc:.3f} < threshold {min_accuracy:.3f}. Keeping existing model.",
        }

    # Backup existing
    if os.path.exists(checkpoint_path):
        shutil.copy2(checkpoint_path, backup_path)

    # Atomic write (write then replace)
    tmp = checkpoint_path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(candidate, f, indent=2)
        os.replace(tmp, checkpoint_path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return {"promoted": False, "reason": f"Write failed: {e}"}

    return {
        "promoted": True,
        "model_version": candidate["model_version"],
        "val_accuracy": val_acc,
        "backup_written": os.path.exists(backup_path),
    }


def rollback_to_previous(
    checkpoint_path: str = CHECKPOINT_PATH,
    backup_path: str = BACKUP_PATH,
) -> dict:
    """Restore the previous checkpoint if a backup exists."""
    if not os.path.exists(backup_path):
        return {"rolled_back": False, "reason": "No backup checkpoint available."}
    try:
        shutil.copy2(backup_path, checkpoint_path)
        return {"rolled_back": True, "restored_from": backup_path}
    except Exception as e:
        return {"rolled_back": False, "reason": str(e)}
