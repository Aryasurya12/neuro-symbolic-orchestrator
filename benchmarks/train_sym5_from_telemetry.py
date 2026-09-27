"""SYM-5: Training script — load telemetry, train candidate, optionally promote.

Usage:
    python benchmarks/train_sym5_from_telemetry.py [--promote]

Flags:
    --promote   Promote candidate if validation passes (default: dry-run only)
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.symbolic.telemetry.repository import TelemetryRepository
from src.symbolic.telemetry.feedback import MIN_TELEMETRY_SAMPLES
from src.symbolic.optihive.telemetry_training import (
    train_candidate_from_telemetry,
    promote_candidate_model,
    CANDIDATE_PATH,
    CHECKPOINT_PATH,
    SYM5_MODEL_VERSION,
)


def main():
    do_promote = "--promote" in sys.argv

    print("=" * 60)
    print("SYM-5: TELEMETRY-DRIVEN MODEL TRAINING")
    print("=" * 60)

    repo = TelemetryRepository()
    total = repo.count()
    print(f"\nTotal telemetry records in store: {total}")

    if total < MIN_TELEMETRY_SAMPLES:
        print(f"\n[SKIP] Insufficient telemetry: {total} records.")
        print(f"       Minimum required: {MIN_TELEMETRY_SAMPLES}")
        print("\nNo model update performed. Existing SYM-4 checkpoint preserved.")
        return

    print(f"\nTraining candidate model from {total} records...")
    result = train_candidate_from_telemetry(repo=repo)

    print(f"\nTraining result:")
    print(f"  Status          : {result['status']}")
    print(f"  Records total   : {result['record_count']}")
    print(f"  Valid records   : {result['valid_count']}")
    print(f"  Model version   : {result.get('model_version', 'N/A')}")
    if result["train_accuracy"] is not None:
        print(f"  Train accuracy  : {result['train_accuracy'] * 100:.2f}%")
    if result["val_accuracy"] is not None:
        print(f"  Val accuracy    : {result['val_accuracy'] * 100:.2f}%")

    if result["status"] != "ok":
        print(f"\n[SKIP] Training did not succeed: {result.get('error', 'unknown')}")
        return

    if not do_promote:
        print(f"\n[DRY RUN] Candidate written to: {CANDIDATE_PATH}")
        print("         Run with --promote to replace production checkpoint.")
        print("\nNo promotion performed. Existing SYM-4 checkpoint preserved.")
        return

    print(f"\nAttempting promotion...")
    promo = promote_candidate_model()

    print(f"\nPromotion result:")
    print(f"  Promoted        : {promo['promoted']}")
    if promo["promoted"]:
        print(f"  New version     : {promo.get('model_version', 'N/A')}")
        print(f"  Val accuracy    : {promo.get('val_accuracy', 'N/A')}")
        print(f"  Backup written  : {promo.get('backup_written', False)}")
        print(f"\nCheckpoint updated: {CHECKPOINT_PATH}")
    else:
        print(f"  Reason          : {promo.get('reason', 'unknown')}")
        print(f"\nExisting model preserved unchanged.")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
