"""SYM-5: Training pipeline and candidate validation tests."""

import pytest
import json
import os

from src.symbolic.telemetry.models import RoutingTelemetry
from src.symbolic.telemetry.repository import TelemetryRepository
from src.symbolic.telemetry.feedback import MIN_TELEMETRY_SAMPLES, CLASS_LABELS
from src.symbolic.optihive.telemetry_training import (
    train_candidate_from_telemetry, TRAINING_SEED
)


def _seed_repo(repo: TelemetryRepository, n: int, solver_set=None):
    """Seed a repository with synthetic valid telemetry records."""
    import random
    random.seed(42)
    solver_options = [["GA", "PSO"], ["Z3"], ["GA", "PSO", "Z3"]]
    for i in range(n):
        solvers = solver_set or solver_options[i % 3]
        repo.insert(RoutingTelemetry(
            resource_scale=float(random.randint(10, 1024)),
            service_scale=random.randint(1, 20),
            budget_tightness=float(random.uniform(1.0, 10.0)),
            provider_count=random.randint(1, 4),
            has_latency_constraint=(i % 3 == 0),
            has_sla_constraint=(i % 5 == 0),
            constraint_density=float(random.randint(0, 4)),
            is_highly_constrained=(i % 4 == 0),
            selected_solvers=solvers,
            routing_mode="learned_high",
            learned_confidence=0.85,
            fallback_used=False,
            total_runtime_ms=float(random.uniform(5.0, 50.0)),
            solver_runtime_ms=float(random.uniform(4.0, 48.0)),
            is_feasible=True,
            hard_constraint_violation=False,
            objective_cost_usd=float(random.uniform(50, 500)),
            model_version="sym4-v1",
        ))


def test_insufficient_data_returns_correct_status(tmp_path):
    repo = TelemetryRepository(db_path=str(tmp_path / "telem.db"))
    _seed_repo(repo, MIN_TELEMETRY_SAMPLES - 1)
    result = train_candidate_from_telemetry(repo=repo)
    assert result["status"] == "insufficient_data"


def test_training_succeeds_with_enough_data(tmp_path):
    repo = TelemetryRepository(db_path=str(tmp_path / "telem.db"))
    _seed_repo(repo, MIN_TELEMETRY_SAMPLES + 10)
    result = train_candidate_from_telemetry(repo=repo)
    assert result["status"] == "ok"
    assert result["train_accuracy"] is not None
    assert result["val_accuracy"] is not None


def test_training_is_deterministic(tmp_path):
    """Two runs with same data must produce the same checkpoint."""
    repo = TelemetryRepository(db_path=str(tmp_path / "telem.db"))
    _seed_repo(repo, MIN_TELEMETRY_SAMPLES + 20)

    r1 = train_candidate_from_telemetry(repo=repo)
    r2 = train_candidate_from_telemetry(repo=repo)

    assert r1["status"] == "ok"
    assert r2["status"] == "ok"
    assert abs(r1["val_accuracy"] - r2["val_accuracy"]) < 1e-9


def test_candidate_checkpoint_written(tmp_path):
    repo = TelemetryRepository(db_path=str(tmp_path / "telem.db"))
    _seed_repo(repo, MIN_TELEMETRY_SAMPLES + 10)
    train_candidate_from_telemetry(repo=repo)

    from src.symbolic.optihive.telemetry_training import CANDIDATE_PATH
    assert os.path.exists(CANDIDATE_PATH)
    with open(CANDIDATE_PATH) as f:
        data = json.load(f)
    assert "W" in data
    assert "model_version" in data


def test_model_version_is_sym5(tmp_path):
    repo = TelemetryRepository(db_path=str(tmp_path / "telem.db"))
    _seed_repo(repo, MIN_TELEMETRY_SAMPLES + 5)
    result = train_candidate_from_telemetry(repo=repo)
    assert result["model_version"] == "sym5-v1"
