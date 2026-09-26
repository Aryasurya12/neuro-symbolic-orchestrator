"""SYM-6 Remediation: Comprehensive Safety Arbitration Test Suite.

Tests all 16 required conditions:
1. Latency constraint forces Z3.
2. SLA constraint forces Z3.
3. DR/topology workload forces Z3.
4. Multi-region/topology workload forces Z3.
5. No topology-sensitive constraint does NOT automatically force Z3.
6. Learned router predicting [GA, PSO] is overridden safely when topology constraints require Z3.
7. SYM-3 [GA, PSO] proposal is overridden safely when required.
8. Learned router failure falls back safely.
9. Safety arbitration failure falls back to [GA, PSO, Z3].
10. Duplicate solvers are removed.
11. Solver ordering remains deterministic.
12. Existing ILP hard-feasibility filtering remains unchanged.
13. Existing infeasible workloads remain infeasible.
14. Feasible DR workload E1_DR_Feasible now reaches Z3.
15. The final result for E1_DR_Feasible is compared against the reference race.
16. No hard constraint violation is introduced.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.symbolic.models import (
    SymbolicOptimizationRequest,
    OptimizationCandidate,
    ConstraintStatus
)
from src.symbolic.optihive.adaptive_router import AdaptiveSolverRouter, SolverRoutingFeatures
from src.symbolic.optihive.safety_arbitration import (
    SafetyArbitrator,
    SafetyArbitrationResult,
    apply_safety_arbitration,
    RULE_TOPOLOGY_REQUIRES_Z3,
    RULE_HIGHLY_CONSTRAINED_Z3,
    RULE_Z3_ALONE_NEEDS_HEURISTIC,
    RULE_EMPTY_FALLBACK
)
from src.symbolic.optihive.ilp_filter import ILPSyntacticFilter
from src.symbolic.optihive.eval_runner import run_sym3, run_sym4, run_reference_race


def _make_req(**kwargs) -> SymbolicOptimizationRequest:
    defaults = dict(
        problem_type="Standard_Compute",
        cloud_providers=["aws", "azure"],
        budget_max_usd=1000.0,
        service_count=3,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=0.0,
        sla_availability_pct=0.0,
    )
    defaults.update(kwargs)
    return SymbolicOptimizationRequest(**defaults)


# ─── Tests 1-5: Constraint triggers ──────────────────────────────────────────

def test_1_latency_constraint_forces_z3():
    arbitrator = SafetyArbitrator()
    req = _make_req(latency_max_ms=50.0)
    res = arbitrator.evaluate(req, ["GA", "PSO"])
    assert "Z3" in res.final_solver_set
    assert res.safety_triggered
    assert RULE_TOPOLOGY_REQUIRES_Z3 in res.rules_applied


def test_2_sla_constraint_forces_z3():
    arbitrator = SafetyArbitrator()
    req = _make_req(sla_availability_pct=99.9)
    res = arbitrator.evaluate(req, ["GA", "PSO"])
    assert "Z3" in res.final_solver_set
    assert res.safety_triggered
    assert RULE_TOPOLOGY_REQUIRES_Z3 in res.rules_applied


def test_3_dr_topology_workload_forces_z3():
    arbitrator = SafetyArbitrator()
    req = _make_req(problem_type="E1_DR_Feasible")
    res = arbitrator.evaluate(req, ["GA", "PSO"])
    assert "Z3" in res.final_solver_set
    assert res.safety_triggered


def test_4_multi_region_topology_workload_forces_z3():
    arbitrator = SafetyArbitrator()
    req = _make_req(problem_type="Multi_Region_Deployment")
    res = arbitrator.evaluate(req, ["GA", "PSO"])
    assert "Z3" in res.final_solver_set
    assert res.safety_triggered


def test_5_no_topology_constraint_does_not_force_z3():
    arbitrator = SafetyArbitrator()
    req = _make_req(
        problem_type="A1_Small",
        cloud_providers=["aws", "azure", "gcp"],
        budget_max_usd=2000.0,
        service_count=2,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=0.0,
        sla_availability_pct=0.0
    )
    res = arbitrator.evaluate(req, ["GA", "PSO"])
    assert "Z3" not in res.final_solver_set
    assert not res.safety_triggered


# ─── Tests 6-7: Router override ──────────────────────────────────────────────

def test_6_learned_router_predicting_ga_pso_overridden_for_topology():
    router = AdaptiveSolverRouter()
    req = _make_req(
        latency_max_ms=30.0,
        sla_availability_pct=99.9,
        problem_type="E1_DR_Feasible"
    )
    decision = router.route(req)
    assert "Z3" in decision.selected_solvers
    assert "Safety arbitration triggered" in decision.reason or "Safety" in decision.reason


def test_7_sym3_ga_pso_proposal_overridden_when_required():
    class DetOnlyRouter(AdaptiveSolverRouter):
        def __init__(self):
            self.learned_router = None

    router = DetOnlyRouter()
    req = _make_req(latency_max_ms=45.0)
    decision = router.route(req)
    assert "Z3" in decision.selected_solvers


# ─── Tests 8-9: Fallback safety ──────────────────────────────────────────────

def test_8_learned_router_failure_falls_back_safely():
    router = AdaptiveSolverRouter()
    if router.learned_router:
        router.learned_router.predict = MagicMock(side_effect=RuntimeError("Model corrupted"))
    req = _make_req(latency_max_ms=25.0)
    decision = router.route(req)
    assert len(decision.selected_solvers) > 0
    assert "Z3" in decision.selected_solvers


def test_9_safety_arbitration_failure_falls_back_to_full_race():
    arbitrator = SafetyArbitrator()
    req = _make_req()
    with patch.object(arbitrator, "_apply_rules", side_effect=Exception("Unexpected crash")):
        res = arbitrator.evaluate(req, ["GA"])
        assert res.final_solver_set == ["GA", "PSO", "Z3"]
        assert res.safety_triggered
        assert RULE_EMPTY_FALLBACK in res.rules_applied


# ─── Tests 10-11: Canonical order & deduplication ────────────────────────────

def test_10_duplicate_solvers_removed():
    arbitrator = SafetyArbitrator()
    req = _make_req()
    res = arbitrator.evaluate(req, ["GA", "GA", "PSO", "PSO", "GA"])
    assert res.final_solver_set == ["GA", "PSO"]
    assert len(res.final_solver_set) == len(set(res.final_solver_set))


def test_11_solver_ordering_remains_deterministic():
    arbitrator = SafetyArbitrator()
    req = _make_req(latency_max_ms=10.0)
    res1 = arbitrator.evaluate(req, ["PSO", "GA"])
    res2 = arbitrator.evaluate(req, ["GA", "PSO"])
    assert res1.final_solver_set == res2.final_solver_set
    assert res1.final_solver_set == ["GA", "PSO", "Z3"]


# ─── Tests 12-13: Hard constraints & infeasibility preservation ──────────────

def test_12_existing_ilp_hard_feasibility_filtering_unchanged():
    req = _make_req(budget_max_usd=100.0)
    cand_over_budget = OptimizationCandidate(
        decision_variables={"vms": [1]},
        objective_cost_usd=150.0,
        is_feasible=True
    )
    fc = ILPSyntacticFilter.filter(cand_over_budget, req)
    assert not fc.syntactically_feasible
    assert any("Budget exceeded" in v for v in fc.violations)


def test_13_existing_infeasible_workloads_remain_infeasible():
    req_impossible = _make_req(
        budget_max_usd=0.01,
        required_vcpus=512,
        required_ram_gb=2048.0
    )
    outcome = run_reference_race("C1_Impossible", req_impossible)
    assert not outcome.is_feasible


# ─── Tests 14-16: E1_DR_Feasible end-to-end verification ─────────────────────

def test_14_feasible_dr_workload_reaches_z3():
    req_e1 = _make_req(
        problem_type="E1_DR_Feasible",
        cloud_providers=["aws", "azure"],
        budget_max_usd=500.0,
        service_count=2,
        required_vcpus=1,
        required_ram_gb=4.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.9
    )
    router = AdaptiveSolverRouter()
    decision = router.route(req_e1)
    assert "Z3" in decision.selected_solvers


def test_15_final_result_for_e1_compared_against_reference_race():
    req_e1 = _make_req(
        problem_type="E1_DR_Feasible",
        cloud_providers=["aws", "azure"],
        budget_max_usd=500.0,
        service_count=2,
        required_vcpus=1,
        required_ram_gb=4.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.9
    )
    sym3_out = run_sym3("E1_DR_Feasible", req_e1)
    sym4_out = run_sym4("E1_DR_Feasible", req_e1)
    ref_out = run_reference_race("E1_DR_Feasible", req_e1)

    assert ref_out.is_feasible, "Reference race must be feasible"
    assert sym3_out.is_feasible, "SYM-3 must now be feasible after safety arbitration"
    assert sym4_out.is_feasible, "SYM-4 must be feasible after safety arbitration"
    assert sym3_out.objective_cost_usd is not None
    assert sym4_out.objective_cost_usd is not None


def test_16_no_hard_constraint_violation_introduced():
    req_e1 = _make_req(
        problem_type="E1_DR_Feasible",
        cloud_providers=["aws", "azure"],
        budget_max_usd=500.0,
        service_count=2,
        required_vcpus=1,
        required_ram_gb=4.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.9
    )
    outcome = run_sym4("E1_DR_Feasible", req_e1)
    assert outcome.is_feasible
    assert not outcome.hard_constraint_violation
    assert outcome.objective_cost_usd <= req_e1.budget_max_usd
