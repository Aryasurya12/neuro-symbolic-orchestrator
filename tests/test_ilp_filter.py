import pytest
from src.symbolic.models import OptimizationCandidate, SymbolicOptimizationRequest, ConstraintStatus
from src.symbolic.optihive.ilp_filter import ILPSyntacticFilter

def _create_mock_req():
    return SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=100.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )

def test_ilp_filter_passes_feasible():
    req = _create_mock_req()
    cand = OptimizationCandidate(
        decision_variables={"provider": "AWS"},
        objective_cost_usd=90.0,
        is_feasible=True,
        constraint_status=ConstraintStatus(
            is_feasible=True, budget_ok=True, vcpu_ok=True, ram_ok=True, latency_ok=True, sla_ok=True
        )
    )
    
    res = ILPSyntacticFilter.filter(cand, req)
    assert res.syntactically_feasible
    assert not res.violations

def test_ilp_filter_rejects_budget_violation():
    req = _create_mock_req()
    cand = OptimizationCandidate(
        decision_variables={"provider": "AWS"},
        objective_cost_usd=150.0, # Breaks budget
        is_feasible=True,
        constraint_status=ConstraintStatus(
            is_feasible=True, budget_ok=True, vcpu_ok=True, ram_ok=True, latency_ok=True, sla_ok=True
        )
    )
    
    res = ILPSyntacticFilter.filter(cand, req)
    assert not res.syntactically_feasible
    assert any("Budget exceeded" in v for v in res.violations)

def test_ilp_filter_rejects_invalid_provider():
    req = _create_mock_req()
    cand = OptimizationCandidate(
        decision_variables={"provider_a": "GCP"}, # GCP not in request
        objective_cost_usd=90.0,
        is_feasible=True,
        constraint_status=ConstraintStatus(
            is_feasible=True, budget_ok=True, vcpu_ok=True, ram_ok=True, latency_ok=True, sla_ok=True
        )
    )
    
    res = ILPSyntacticFilter.filter(cand, req)
    assert not res.syntactically_feasible
    assert any("Invalid provider selected" in v for v in res.violations)

def test_ilp_filter_rejects_constraint_status_failure():
    req = _create_mock_req()
    cand = OptimizationCandidate(
        decision_variables={"provider": "AWS"},
        objective_cost_usd=90.0,
        is_feasible=False,
        constraint_status=ConstraintStatus(
            is_feasible=False, budget_ok=True, vcpu_ok=False, ram_ok=True, latency_ok=True, sla_ok=True
        )
    )
    
    res = ILPSyntacticFilter.filter(cand, req)
    assert not res.syntactically_feasible
    assert any("vCPU violated" in v for v in res.violations)
