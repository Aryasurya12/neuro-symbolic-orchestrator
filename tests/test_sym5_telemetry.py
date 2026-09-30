"""SYM-5: Telemetry model and persistence tests."""

import pytest
import os
from src.symbolic.telemetry.models import RoutingTelemetry
from src.symbolic.telemetry.repository import TelemetryRepository


@pytest.fixture
def tmp_repo(tmp_path):
    db = str(tmp_path / "test_telemetry.db")
    return TelemetryRepository(db_path=db)


def make_record(**kwargs) -> RoutingTelemetry:
    defaults = dict(
        resource_scale=64.0,
        service_scale=4,
        budget_tightness=5.0,
        provider_count=2,
        has_latency_constraint=False,
        has_sla_constraint=False,
        constraint_density=1.0,
        is_highly_constrained=False,
        selected_solvers=["GA", "PSO"],
        routing_mode="learned_high",
        learned_confidence=0.92,
        fallback_used=False,
        total_runtime_ms=12.0,
        solver_runtime_ms=10.0,
        is_feasible=True,
        hard_constraint_violation=False,
        objective_cost_usd=120.0,
        model_version="sym4-v1",
    )
    defaults.update(kwargs)
    return RoutingTelemetry(**defaults)


def test_telemetry_model_creation():
    r = make_record()
    assert r.resource_scale == 64.0
    assert r.is_feasible is True
    assert "GA" in r.selected_solvers


def test_telemetry_model_defaults():
    r = RoutingTelemetry()
    assert r.run_id != ""
    assert r.timestamp > 0
    assert r.selected_solvers == []


def test_telemetry_persistence(tmp_repo):
    r = make_record()
    tmp_repo.insert(r)
    assert tmp_repo.count() == 1


def test_telemetry_retrieval(tmp_repo):
    r = make_record(resource_scale=128.0)
    tmp_repo.insert(r)
    records = tmp_repo.fetch_all()
    assert len(records) == 1
    assert records[0].resource_scale == 128.0


def test_telemetry_serializes_solver_list(tmp_repo):
    r = make_record(selected_solvers=["GA", "PSO", "Z3"])
    tmp_repo.insert(r)
    fetched = tmp_repo.fetch_all()[0]
    assert fetched.selected_solvers == ["GA", "PSO", "Z3"]


def test_telemetry_multiple_records(tmp_repo):
    for i in range(5):
        tmp_repo.insert(make_record(resource_scale=float(i * 10)))
    assert tmp_repo.count() == 5
    records = tmp_repo.fetch_all()
    assert len(records) == 5


def test_telemetry_fetch_recent(tmp_repo):
    for i in range(10):
        tmp_repo.insert(make_record())
    recent = tmp_repo.fetch_recent(limit=3)
    assert len(recent) == 3
