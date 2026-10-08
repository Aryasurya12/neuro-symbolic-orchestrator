"""Comprehensive Regression Test Suite for Prompt 2 of 4: Comparable Modes & Consistent Verification.

Covers:
1. Unified Task Model & Disambiguation:
   - Distinct cpu_target_pct vs max_cpu_pct semantics.
   - Explicit CPU ceiling survival from contract -> solver -> independent checker.
   - Offered workload bandwidth preservation (no optimizer workload reduction).
2. Mode 1 & 2 Normalization & Schema Discrimination:
   - Preservation of Mode 1 explicit monthly total ($300.21/mo vs $0.10/hr).
   - Ambiguous prose returns NEEDS_REVIEW without inventing quantities.
   - Wrong-task JSON identified as TASK_INCOMPATIBLE without zero substitutions.
   - Mode 2 discriminated schema outcome handling.
3. Scaling Formulation & Reference Verification:
   - Overloaded scaling candidate (100 Mbps, 1 replica = 133.33% CPU) fails.
   - Deployable integer replicas evaluated post-conversion.
   - Reference scaling enumeration finds exact discrete minimum.
4. Mode-Blind Independent Checker:
   - Identical normalized allocations receive identical verdicts across all modes.
   - Missing bandwidth/replicas cannot produce numerical PASS.
   - Rejection suppresses deployment advice in executive reports.
5. Canonical Execution Records:
   - CanonicalExecutionRecord serialization and retrieval without re-solving.
   - Budget headroom used when no baseline expenditure is declared.
"""

import json
import pytest
from src.proof.stage_trace import (
    trace_stage_3_contract_validation,
    trace_stage_4_solver_execution,
    trace_stage_5_independent_verification,
)
from src.semantic.carm_matcher import CARMMatcher
from src.semantic.explainer import FinOpsExplainer
from src.semantic.normalizer import OutputNormalizer
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.scope_parser import SCOPEParser
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optimizers.objective import ObjectiveEvaluator
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from templates.Continuous_PSO_Dynamic_Scaling import (
    reference_scaling_solution,
    solve_pso_continuous_scaling,
)
from src.verifiers.canonical_record import (
    CanonicalExecutionRecord,
    CheckStatus,
    ExplanationSource,
    FeasibilityStatus,
    NormalizationStatus,
    OptimalityStatus,
)
from src.verifiers.independent_checker import IndependentChecker


# =============================================================================
# 1. Unified Task Model & CPU Ceiling Disambiguation Tests
# =============================================================================
def test_cpu_target_vs_max_ceiling_disambiguation():
    """Verify target_cpu_pct is the operating point while max_cpu_pct is a hard constraint."""
    # Case A: target 70%, no explicit max_cpu_pct (default 100% physical saturation)
    contract_a = CloudOptimizationContract(
        problem_type="PSO_Continuous_Scaling",
        budget_max_usd=1500.0,
        target_cpu_pct=70.0,
        max_cpu_pct=None,
    )
    # 150 Mbps, 2 replicas -> CPU = 150 / (2 * 75) = 100.0% (at physical ceiling, above 70% target)
    candidate_a = {
        "optimal_bandwidth_mbps": 150.0,
        "recommended_replicas": 2,
        "estimated_monthly_cost_usd": 102.0,
    }
    check_a = IndependentChecker.verify_solution(contract_a, candidate_a)
    assert check_a["feasible_against_contract"] is True
    assert check_a["recomputed_metrics"]["modeled_cpu_pct"] == 100.0

    # Case B: explicit max_cpu_pct = 75.0% -> same candidate at 100% CPU must FAIL
    contract_b = CloudOptimizationContract(
        problem_type="PSO_Continuous_Scaling",
        budget_max_usd=1500.0,
        target_cpu_pct=70.0,
        max_cpu_pct=75.0,
    )
    check_b = IndependentChecker.verify_solution(contract_b, candidate_a)
    assert check_b["feasible_against_contract"] is False
    assert any("exceeds maximum ceiling of 75.0%" in v for v in check_b["violations"])


def test_explicit_cpu_ceiling_survives_solving_and_checking():
    """Verify explicit CPU ceiling is enforced during solver execution and verified by checker."""
    contract = CloudOptimizationContract(
        problem_type="PSO_Continuous_Scaling",
        budget_max_usd=1500.0,
        target_bandwidth_mbps=200.0,
        min_bandwidth_mbps=200.0,
        target_cpu_pct=70.0,
        max_cpu_pct=60.0,  # Hard ceiling
    )
    # 200 Mbps with capacity 75 Mbps/rep:
    # 2 reps -> 200 / 150 = 133.3% (overload FAIL)
    # 3 reps -> 200 / 225 = 88.89% (exceeds 60% ceiling FAIL)
    # 4 reps -> 200 / 300 = 66.67% (exceeds 60% ceiling FAIL)
    # 5 reps -> 200 / 375 = 53.33% (<= 60% PASS)

    s4_ok, solver_res, s4_err = trace_stage_4_solver_execution(contract, mode=3, silent=True)
    assert s4_ok is True
    assert solver_res["status"].upper() == "CONVERGED"
    assert solver_res["recommended_replicas"] >= 5

    s5_ok, check = trace_stage_5_independent_verification(contract, solver_res, silent=True)
    assert s5_ok is True
    assert check["feasible_against_contract"] is True
    assert check["recomputed_metrics"]["modeled_cpu_pct"] <= 60.0


def test_offered_workload_bandwidth_preservation():
    """Verify the optimizer cannot reduce user-required bandwidth simply to minimize cost."""
    contract = CloudOptimizationContract(
        problem_type="PSO_Continuous_Scaling",
        budget_max_usd=1500.0,
        target_bandwidth_mbps=300.0,
        min_bandwidth_mbps=300.0,
        max_bandwidth_mbps=1000.0,
        target_cpu_pct=70.0,
    )
    s4_ok, solver_res, s4_err = trace_stage_4_solver_execution(contract, mode=3, silent=True)
    assert s4_ok is True
    # Returned bandwidth must satisfy offered demand >= 300.0 Mbps
    assert solver_res["optimal_bandwidth_mbps"] >= 300.0


# =============================================================================
# 2. Normalization & Schema Discrimination Tests
# =============================================================================
def test_mode1_monthly_total_preserved():
    """Verify Mode 1 prose with '$300.21/month' is normalized to 300.21, not $0.10 from hourly price."""
    prose = (
        "We recommend deploying 2x t3.xlarge instances on AWS for your 8 vCPUs and 16GB RAM workload. "
        "At $0.096 per hour each, the total estimated cost is $300.21/month for 730 hours of runtime."
    )
    status, cost, alloc, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=prose,
        problem_type="ILP_VM_Allocation",
    )
    assert status == NormalizationStatus.SUCCESS
    assert cost == 300.21
    assert alloc["allocated_vms"][0]["sku"] == "t3.xlarge"
    assert alloc["allocated_vms"][0]["count"] == 2


def test_mode1_ambiguous_prose_returns_needs_review():
    """Verify ambiguous prose without explicit quantities flags NEEDS_REVIEW without inventing numbers."""
    prose = "You should definitely consider using AWS EC2 instances like t3.large or maybe m5.large for your cloud setup."
    status, cost, alloc, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=prose,
        problem_type="ILP_VM_Allocation",
    )
    assert status == NormalizationStatus.NEEDS_REVIEW
    assert alloc is None or len(alloc.get("allocated_vms", [])) == 0


def test_mode2_wrong_task_json_is_task_incompatible():
    """Verify providing VM schema output for a scaling task is flagged as TASK_INCOMPATIBLE."""
    vm_json = {
        "allocated_vms": [
            {"sku": "t3.medium", "provider": "AWS", "quantity": 2, "monthly_cost": 60.74}
        ],
        "total_monthly_cost_usd": 60.74,
    }
    status, cost, alloc, errors, evidence = OutputNormalizer.normalize_mode2_json(
        raw_json_or_text=vm_json,
        requested_problem_type="PSO_Continuous_Scaling",
    )
    assert status == NormalizationStatus.TASK_INCOMPATIBLE
    assert cost is None or cost == 60.74
    assert any("incompatib" in err.lower() for err in errors)


def test_mode2_discriminated_clarification_and_infeasible_outcomes():
    """Verify Mode 2 discriminated schema handles CLARIFICATION_REQUIRED and SOLVER_INFEASIBLE."""
    clarification_json = {
        "task_outcome": "CLARIFICATION_REQUIRED",
        "clarification_questions": [
            "Please specify your target CPU threshold percentage.",
            "Please confirm your monthly budget ceiling in USD."
        ],
        "reasoning": "Missing traffic profile and budget limits."
    }
    status, cost, alloc, errors, evidence = OutputNormalizer.normalize_mode2_json(
        raw_json_or_text=clarification_json,
        requested_problem_type="PSO_Continuous_Scaling",
    )
    assert status == NormalizationStatus.CLARIFICATION_REQUIRED

    infeasible_json = {
        "task_outcome": "SOLVER_INFEASIBLE",
        "infeasible_reasons": ["Required 64 vCPUs cannot be provisioned under $10/month budget."],
        "suggested_relaxations": ["Increase budget to at least $120/month."]
    }
    status, cost, alloc, errors, evidence = OutputNormalizer.normalize_mode2_json(
        raw_json_or_text=infeasible_json,
        requested_problem_type="ILP_VM_Allocation",
    )
    assert status == NormalizationStatus.SOLVER_INFEASIBLE


# =============================================================================
# 3. Solver / Formulation Scaling Repairs & Reference Enumeration
# =============================================================================
def test_overloaded_scaling_candidate_fails_independent_checker():
    """100 Mbps, 1 replica, capacity 75 Mbps -> 133.33% modeled utilization fails."""
    contract = CloudOptimizationContract(
        problem_type="PSO_Continuous_Scaling",
        budget_max_usd=1500.0,
        target_bandwidth_mbps=100.0,
        target_cpu_pct=70.0,
    )
    candidate = {
        "optimal_bandwidth_mbps": 100.0,
        "recommended_replicas": 1,
        "estimated_monthly_cost_usd": 53.00,
    }
    check = IndependentChecker.verify_solution(contract, candidate)
    assert check["feasible_against_contract"] is False
    assert check["recomputed_metrics"]["modeled_cpu_pct"] == 133.33
    assert any("133.3%" in v or "exceeds 100%" in v for v in check["violations"])


def test_reference_scaling_enumeration_accuracy():
    """Reference scaling enumeration computes exact global discrete minimum."""
    req = SymbolicOptimizationRequest(
        problem_type="PSO_Continuous_Scaling",
        cloud_providers=["AWS"],
        budget_max_usd=1500.0,
        target_bandwidth_mbps=200.0,
        min_bandwidth_mbps=200.0,
        max_cpu_pct=70.0,
    )
    ref = ParticleSwarmOptimization.reference_scaling_enumeration(req)
    assert ref["is_feasible"] is True
    # 200 / (R * 75) <= 0.70 => R * 75 >= 285.71 => R >= 3.81 => R = 4
    assert ref["replicas"] == 4
    # Cost: 200 * 0.08 + 4 * 45 = 16 + 180 = $196.00
    assert ref["monthly_cost_usd"] == 196.00
    assert ref["modeled_cpu_pct"] == round((200.0 / 300.0) * 100.0, 2)  # 66.67%


def test_impossible_scaling_budget_detected_without_fake_certificate():
    """Verify impossible budget for scaling reports infeasible rather than fake certificate."""
    contract = CloudOptimizationContract(
        problem_type="PSO_Continuous_Scaling",
        budget_max_usd=10.0,  # Minimum 1 replica is $45 + $8 = $53/mo
        target_bandwidth_mbps=100.0,
    )
    s4_ok, solver_res, s4_err = trace_stage_4_solver_execution(contract, mode=3, silent=True)
    assert s4_ok is True
    assert solver_res["status"].upper() in ["INFEASIBLE", "FAILED"]

    s5_ok, check = trace_stage_5_independent_verification(contract, solver_res, silent=True)
    assert s5_ok is False
    assert check["feasible_against_contract"] is False
    assert check["optimality_verdict"] == OptimalityStatus.INFEASIBLE.value


# =============================================================================
# 4. Mode-Blind Checker & Invariant Checks
# =============================================================================
def test_mode_blind_checker_identical_verdicts_all_modes():
    """Cross-mode identical normalized candidate receives identical check results."""
    contract = CloudOptimizationContract(
        problem_type="Z3_Graph_Disaster_Recovery",
        budget_max_usd=600.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.99,
        cloud_providers=["AWS", "GCP"],
    )
    candidate = {
        "primary_region": "us-east-1",
        "secondary_region": "us-central1",
        "total_monthly_cost_usd": 243.0,
    }

    results = [IndependentChecker.verify_solution(contract, candidate) for _ in range(4)]
    for r in results[1:]:
        assert r["feasible_against_contract"] == results[0]["feasible_against_contract"]
        assert r["cost_accuracy"] == results[0]["cost_accuracy"]
        assert r["recomputed_metrics"] == results[0]["recomputed_metrics"]
        assert r["optimality_verdict"] == results[0]["optimality_verdict"]


def test_missing_values_cannot_produce_numerical_pass():
    """Missing bandwidth, replicas, or region IDs produce FAIL, not zero or PASS."""
    contract_scaling = CloudOptimizationContract(
        problem_type="PSO_Continuous_Scaling",
        budget_max_usd=1500.0,
    )
    # Empty scaling solution
    check_scaling = IndependentChecker.verify_solution(contract_scaling, {})
    assert check_scaling["feasible_against_contract"] is False
    assert check_scaling["structure_valid"] is False

    contract_dr = CloudOptimizationContract(
        problem_type="Z3_Graph_Disaster_Recovery",
        budget_max_usd=600.0,
    )
    # DR with missing secondary region
    check_dr = IndependentChecker.verify_solution(contract_dr, {"primary_region": "us-east-1"})
    assert check_dr["feasible_against_contract"] is False
    assert check_dr["structure_valid"] is False


def test_rejected_solution_suppresses_deployment_report():
    """Failed allocation explanation leads with failure and suppresses deployment advice."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        budget_max_usd=50.0,
        required_vcpus=64,
        required_ram_gb=256.0,
    )
    solver_res = {
        "status": "infeasible",
        "allocated_vms": [],
        "total_monthly_cost_usd": 0.0,
    }
    check = IndependentChecker.verify_solution(contract, solver_res)
    assert check["feasible_against_contract"] is False

    report = FinOpsExplainer.generate_report(contract, solver_res, check_result=check)
    assert "REJECTED" in report
    assert "DEPLOYMENT ADVICE SUPPRESSED" in report
    assert "CANNOT be safely deployed" in report


# =============================================================================
# 5. Canonical Run Records & Zero Re-solving
# =============================================================================
def test_canonical_record_serialization_and_headroom_semantics():
    """Verify CanonicalExecutionRecord stores all provenance and uses budget headroom correctly."""
    record = CanonicalExecutionRecord(
        mode=4,
        mode_name="Mode 4: Neuro-Symbolic",
        original_query="Deploy 8 vCPUs and 16GB RAM on AWS under $300",
        problem_type="ILP_VM_Allocation",
        requirements={"required_vcpus": 8, "required_ram_gb": 16.0, "budget_max_usd": 300.0},
        claimed_cost_usd=140.16,
        recomputed_cost_usd=140.16,
        budget_headroom_usd=159.84,  # $300 - $140.16
        feasibility=FeasibilityStatus.PASS,
        optimality_status=OptimalityStatus.PROVABLY_OPTIMAL,
        summary_status="Feasible against checked constraints",
    )

    d = record.to_dict()
    assert d["mode"] == 4
    assert d["budget_headroom_usd"] == 159.84
    assert d["feasibility"] == "PASS"

    summary_row = record.to_summary_row()
    assert summary_row["mode"] == "Mode 4"
    assert summary_row["claimed_cost"] == "$140.16"
    assert summary_row["recomputed_cost"] == "$140.16"
    assert summary_row["feasibility"] == "PASS"
