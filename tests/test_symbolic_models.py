import pytest
from src.symbolic.models import (
    SymbolicOptimizationRequest, 
    OptimizationCandidate, 
    ConstraintStatus,
    OptimizationMetrics,
    OptimizationResult
)

def test_symbolic_request_creation():
    req = SymbolicOptimizationRequest(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=100.0,
        service_count=1,
        required_vcpus=2,
        required_ram_gb=4.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.9
    )
    assert req.budget_max_usd == 100.0
    assert req.problem_type == "ILP_VM_Allocation"

def test_constraint_status():
    status = ConstraintStatus(
        is_feasible=True,
        budget_ok=True,
        vcpu_ok=True,
        ram_ok=True,
        latency_ok=True,
        sla_ok=True,
        details="All constraints satisfied"
    )
    assert status.is_feasible
    assert status.details == "All constraints satisfied"

def test_candidate_creation():
    cand = OptimizationCandidate(
        decision_variables={"vm_count": 2},
        objective_cost_usd=50.0,
        is_feasible=True
    )
    assert cand.is_feasible
    assert cand.decision_variables["vm_count"] == 2
    assert cand.objective_cost_usd == 50.0

def test_result_creation():
    res = OptimizationResult(
        best_candidate=None,
        is_feasible=False,
        metrics=OptimizationMetrics(runtime_ms=10.0),
        solver_name="TestSolver"
    )
    assert res.solver_name == "TestSolver"
    assert not res.is_feasible
    assert res.metrics.runtime_ms == 10.0
