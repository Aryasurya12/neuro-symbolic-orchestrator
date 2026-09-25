"""SYM-5: Feedback signal tests — label derivation and dataset building."""

import pytest
from src.symbolic.telemetry.models import RoutingTelemetry
from src.symbolic.telemetry.feedback import (
    derive_label, filter_valid_records, build_training_dataset,
    LABEL_GA_PSO, LABEL_Z3, LABEL_GA_PSO_Z3, MIN_TELEMETRY_SAMPLES, CLASS_LABELS
)


def make_record(**kwargs) -> RoutingTelemetry:
    defaults = dict(
        resource_scale=64.0, service_scale=4, budget_tightness=5.0,
        provider_count=2, has_latency_constraint=False, has_sla_constraint=False,
        constraint_density=1.0, is_highly_constrained=False,
        selected_solvers=["GA", "PSO"], routing_mode="learned_high",
        learned_confidence=0.92, fallback_used=False,
        total_runtime_ms=12.0, solver_runtime_ms=10.0,
        is_feasible=True, hard_constraint_violation=False,
        objective_cost_usd=120.0,
    )
    defaults.update(kwargs)
    return RoutingTelemetry(**defaults)


def test_label_ga_pso():
    r = make_record(selected_solvers=["GA", "PSO"])
    assert derive_label(r) == LABEL_GA_PSO


def test_label_z3_only():
    r = make_record(selected_solvers=["Z3"])
    assert derive_label(r) == LABEL_Z3


def test_label_ga_pso_z3():
    r = make_record(selected_solvers=["GA", "PSO", "Z3"])
    assert derive_label(r) == LABEL_GA_PSO_Z3


def test_label_z3_plus_ga():
    r = make_record(selected_solvers=["GA", "Z3"])
    assert derive_label(r) == LABEL_GA_PSO_Z3


def test_label_returns_none_for_zero_runtime():
    r = make_record(selected_solvers=["GA"], total_runtime_ms=0.0)
    assert derive_label(r) is None


def test_filter_removes_violations():
    records = [
        make_record(hard_constraint_violation=True),
        make_record(hard_constraint_violation=False),
    ]
    valid = filter_valid_records(records)
    assert len(valid) == 1
    assert valid[0].hard_constraint_violation is False


def test_filter_removes_zero_runtime():
    records = [
        make_record(total_runtime_ms=0.0),
        make_record(total_runtime_ms=5.0),
    ]
    valid = filter_valid_records(records)
    assert len(valid) == 1


def test_filter_removes_no_solvers():
    records = [
        make_record(selected_solvers=[]),
        make_record(selected_solvers=["GA"]),
    ]
    valid = filter_valid_records(records)
    assert len(valid) == 1


def test_feasible_feedback():
    r = make_record(is_feasible=True, hard_constraint_violation=False)
    label = derive_label(r)
    assert label in CLASS_LABELS


def test_infeasible_feedback_is_still_labeled():
    # Infeasible but no hard constraint violation — still usable for routing learning
    r = make_record(is_feasible=False, hard_constraint_violation=False, selected_solvers=["Z3"])
    label = derive_label(r)
    assert label == LABEL_Z3


def test_insufficient_data_returns_none():
    records = [make_record() for _ in range(MIN_TELEMETRY_SAMPLES - 1)]
    X, y, labels = build_training_dataset(records)
    assert X is None
    assert y is None
    assert labels is None


def test_sufficient_data_returns_dataset():
    records = [make_record() for _ in range(MIN_TELEMETRY_SAMPLES + 5)]
    X, y, labels = build_training_dataset(records)
    assert X is not None
    assert y is not None
    assert len(X) >= MIN_TELEMETRY_SAMPLES
    assert labels == CLASS_LABELS
