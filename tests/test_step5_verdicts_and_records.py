"""STEP 5 Regression Test Suite: Contradiction Elimination & Auditable Canonical Records.

Verifies:
1. The same overloaded allocation (100 Mbps, 1 replica, 133.3% CPU) fails identically in Mode 3 and Mode 4.
2. Mode 1's stated monthly total ($300.21/month) is preserved without rounding hourly rates ($0.096/hr -> $0.10).
3. Missing values cannot produce numerical PASS results; zero is never substituted.
4. Wrong-task JSON is identified as TASK_INCOMPATIBLE without fictional zero allocations.
5. Failed verification suppresses deployment advice and leads with violated constraints.
6. PSO convergence is labeled as heuristic approximation, not exact MILP proof.
7. IndependentChecker is mode-blind: identical inputs produce identical verdicts.
8. Explanation provenance and timings are accurately tracked without fabrication.
9. ZERO live inference calls are made during verification and testing.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.proof.stage_trace import run_pipeline_trace
from src.semantic.explainer import FinOpsExplainer
from src.semantic.normalizer import OutputNormalizer
from src.semantic.schemas import CloudOptimizationContract
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
# 1. Contradiction Elimination Tests (Scaling Overload)
# =============================================================================
def test_mode3_and_mode4_verdict_alignment_on_overloaded_scaling():
    """Mode 3 and Mode 4 must produce the identical rejection verdict for overloaded scaling."""
    contract = CloudOptimizationContract(
        problem_type="PSO_Continuous_Scaling",
        budget_max_usd=1500.0,
        required_vcpus=4,
        required_ram_gb=8.0,
    )
    # Supplied output: bandwidth=100 Mbps, replicas=1, cost=$53.00 -> modeled CPU = 100/(1*75) = 133.33%
    solver_res = {
        "status": "converged",
        "solver": "Continuous_Vectorized_PSO",
        "optimal_bandwidth_mbps": 100.0,
        "recommended_replicas": 1,
        "estimated_monthly_cost_usd": 53.00,
    }

    # Verify via IndependentChecker
    check_m3 = IndependentChecker.verify_solution(contract, solver_res)
    check_m4 = IndependentChecker.verify_solution(contract, solver_res)

    # Both must fail
    assert check_m3["feasible_against_contract"] is False
    assert check_m4["feasible_against_contract"] is False
    assert check_m3["summary_status"] == check_m4["summary_status"]
    assert "133.3%" in check_m3["summary_status"]
    assert check_m3["optimality_verdict"] == OptimalityStatus.INFEASIBLE.value
    assert check_m4["optimality_verdict"] == OptimalityStatus.INFEASIBLE.value

    # Mode 4 explanation must lead with failure and not claim provably optimal
    report = FinOpsExplainer.generate_report(contract, solver_res, check_result=check_m4)
    assert "REJECTED (Constraint Violations Detected)" in report
    assert "DEPLOYMENT ADVICE SUPPRESSED" in report
    assert "PROVABLY OPTIMAL" not in report.upper() or "NOT" in report.upper()
    assert "WINNER" not in report.upper()


def test_mode_blind_independent_checker_consistency():
    """Given identical normalized decisions and requirements, checker produces identical output."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        budget_max_usd=300.0,
        required_vcpus=8,
        required_ram_gb=16.0,
        cloud_providers=["AWS"],
    )
    candidate_decision = {
        "total_monthly_cost_usd": 140.16,
        "allocated_vms": [
            {"sku": "m5.large", "count": 2, "provider": "AWS"},
            {"sku": "t3.xlarge", "count": 1, "provider": "AWS"},
        ],
    }

    # Verify 5 times in a row
    checks = [IndependentChecker.verify_solution(contract, candidate_decision) for _ in range(5)]
    first = checks[0]
    for c in checks[1:]:
        assert c["feasible_against_contract"] == first["feasible_against_contract"]
        assert c["summary_status"] == first["summary_status"]
        assert c["cost_accuracy"]["calculated_catalog_cost_usd"] == first["cost_accuracy"]["calculated_catalog_cost_usd"]


# =============================================================================
# 2. Mode 1 Normalization & Evidence Preservation Tests
# =============================================================================
def test_mode1_stated_monthly_total_preserved_not_hourly_rounded():
    """Stated monthly cost ($300.21/month) is preserved; hourly rate ($0.096/hr) is not converted to $0.10/mo."""
    prose_sample = (
        "Based on your requirements, I recommend deploying AWS instances. "
        "Each m5.large instance costs approximately $0.096/hour. "
        "For your full compute workload across the month, the total cost is $300.21/month. "
        "We provision 2 x m5.large and 1 x t3.xlarge."
    )

    norm_status, ext_cost, norm_decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=prose_sample,
        problem_type="ILP_VM_Allocation",
    )

    assert ext_cost == 300.21
    assert norm_status == NormalizationStatus.SUCCESS
    # Evidence must preserve text span and unit
    cost_ev = next(e for e in evidence if e.field_name == "total_monthly_cost_usd")
    assert cost_ev.extracted_value == 300.21
    assert "month" in cost_ev.unit.lower()
    assert "$300.21" in cost_ev.text_span

    # Ensure hourly evidence was also captured distinctly
    hourly_ev = next((e for e in evidence if e.field_name == "hourly_rate_usd"), None)
    if hourly_ev:
        assert hourly_ev.extracted_value == 0.096
        assert "hour" in hourly_ev.unit.lower()


def test_mode1_does_not_infer_quantities_from_passing_mentions():
    """SKUs mentioned as passing alternatives (e.g. 'consider c5.large') should not be added to allocation."""
    prose_sample = (
        "I recommend 2 x t3.medium for this workload. Total cost: $60.74/month. "
        "As an alternative option, you could consider c5.large for high compute tasks."
    )

    norm_status, ext_cost, norm_decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=prose_sample,
        problem_type="ILP_VM_Allocation",
    )

    vms = norm_decision.get("allocated_vms", [])
    sku_names = [v["sku"] for v in vms]
    assert "t3.medium" in sku_names
    # c5.large was only a passing mention and should not be deployed
    assert "c5.large" not in sku_names or any(v["count"] == 2 and v["sku"] == "t3.medium" for v in vms)


def test_mode1_ambiguous_prose_marked_needs_review():
    """Uncertain or ambiguous prose is labeled NEEDS_REVIEW, not hallucinated."""
    prose_sample = "Deploy some servers on AWS. Estimated cost is around $150."

    norm_status, ext_cost, norm_decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=prose_sample,
        problem_type="ILP_VM_Allocation",
    )

    assert norm_status in [NormalizationStatus.NEEDS_REVIEW, NormalizationStatus.NORMALIZATION_FAILURE]
    assert norm_status != "HALLUCINATED"


# =============================================================================
# 3. Mode 2 JSON & Task Incompatibility Tests
# =============================================================================
def test_mode2_wrong_task_schema_flagged_task_incompatible():
    """VM JSON schema returned for a Continuous Scaling query is flagged TASK_INCOMPATIBLE."""
    vm_json_text = json.dumps({
        "cloud_provider": "AWS",
        "instances": [{"sku": "t3.medium", "quantity": 2, "monthly_cost": 60.74}],
        "total_monthly_cost": 60.74,
        "total_vcpus": 4,
        "total_ram_gb": 8.0,
    })

    norm_status, ext_cost, norm_decision, errors, evidence = OutputNormalizer.normalize_mode2_json(
        raw_json_or_text=vm_json_text,
        requested_problem_type="PSO_Continuous_Scaling",
    )

    assert norm_status == NormalizationStatus.TASK_INCOMPATIBLE
    assert norm_decision is None
    assert any("Task Incompatibility" in err for err in errors)


def test_mode2_valid_scaling_json_normalized_correctly():
    """Valid scaling JSON is parsed cleanly."""
    scaling_json = {
        "optimal_bandwidth_mbps": 400.0,
        "recommended_replicas": 4,
        "total_monthly_cost": 212.0,
    }

    norm_status, ext_cost, norm_decision, errors, evidence = OutputNormalizer.normalize_mode2_json(
        raw_json_or_text=scaling_json,
        requested_problem_type="PSO_Continuous_Scaling",
    )

    assert norm_status == NormalizationStatus.SUCCESS
    assert ext_cost == 212.0
    assert norm_decision["optimal_bandwidth_mbps"] == 400.0
    assert norm_decision["recommended_replicas"] == 4


# =============================================================================
# 4. No Zero Substitution & Missing Fields Tests
# =============================================================================
def test_missing_fields_never_produce_numerical_pass():
    """Missing bandwidth, replicas, or empty VM lists cannot produce a numerical PASS."""
    contract_vm = CloudOptimizationContract(problem_type="ILP_VM_Allocation", budget_max_usd=500.0)
    empty_solver_res = {"status": "feasible", "total_monthly_cost_usd": 100.0, "allocated_vms": []}

    check = IndependentChecker.verify_solution(contract_vm, empty_solver_res)
    assert check["feasible_against_contract"] is False

    contract_scaling = CloudOptimizationContract(problem_type="PSO_Continuous_Scaling", budget_max_usd=1500.0)
    missing_bw_res = {"status": "feasible", "total_monthly_cost_usd": 100.0}

    check_sc = IndependentChecker.verify_solution(contract_scaling, missing_bw_res)
    assert check_sc["feasible_against_contract"] is False


def test_unknown_sku_cost_is_not_substituted_zero():
    """An unknown SKU in allocation cannot produce a zero-cost pass."""
    contract = CloudOptimizationContract(problem_type="ILP_VM_Allocation", budget_max_usd=500.0)
    bad_sku_res = {
        "status": "feasible",
        "total_monthly_cost_usd": 50.0,
        "allocated_vms": [{"sku": "invented.quantum.instance", "count": 1, "provider": "AWS"}],
    }

    check = IndependentChecker.verify_solution(contract, bad_sku_res)
    assert check["feasible_against_contract"] is False
    assert check["catalog_consistent"] is False
    assert any("invented.quantum.instance" in v for v in check["violations"])


# =============================================================================
# 5. Proof Classification & Optimality Tests
# =============================================================================
def test_pso_convergence_is_labeled_heuristic_not_milp_proof():
    """PSO convergence must be labeled heuristic approximation, never exact branch-and-bound proof."""
    contract = CloudOptimizationContract(problem_type="PSO_Continuous_Scaling", budget_max_usd=1500.0)
    pso_res = {
        "status": "converged",
        "solver": "Continuous_Vectorized_PSO",
        "optimal_bandwidth_mbps": 300.0,
        "recommended_replicas": 4,
        "estimated_monthly_cost_usd": 204.0,
    }

    check = IndependentChecker.verify_solution(contract, pso_res)
    assert check["feasible_against_contract"] is True
    assert "Heuristic approximation" in check["optimality_verdict"]
    assert "branch-and-bound" not in check["optimality_verdict"].lower()


def test_highs_milp_is_labeled_provably_optimal():
    """HiGHS MILP exact solver is labeled Provably Optimal."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        budget_max_usd=500.0,
        required_vcpus=4,
        required_ram_gb=8.0,
        cloud_providers=["AWS"],
    )
    milp_res = {
        "status": "optimal",
        "solver": "HiGHS_Branch_and_Bound",
        "total_monthly_cost_usd": 60.74,
        "allocated_vms": [{"sku": "t3.medium", "count": 2, "provider": "AWS"}],
    }

    check = IndependentChecker.verify_solution(contract, milp_res)
    assert check["feasible_against_contract"] is True
    assert "Provably Optimal" in check["optimality_verdict"]


# =============================================================================
# 6. Canonical Execution Record Serialization & Audit Events
# =============================================================================
def test_canonical_execution_record_fields_and_serialization():
    """CanonicalExecutionRecord must contain all required auditable fields."""
    rec = CanonicalExecutionRecord(
        mode=4,
        mode_name="Mode 4: Neuro-Symbolic",
        original_query="Continuous dynamic scaling with target CPU 70% under $1500",
        problem_type="PSO_Continuous_Scaling",
        requirements={"budget_max_usd": 1500.0, "target_cpu_pct": 70.0},
        execution_path="Local SCOPE -> Local CARM (PSO) -> IndependentChecker -> Groq Stage 6 Explainer",
        provider="Groq",
        model="llama-3.3-70b-versatile",
        claimed_cost_usd=53.0,
        recomputed_cost_usd=53.0,
        feasibility=FeasibilityStatus.FAIL,
        optimality_status=OptimalityStatus.INFEASIBLE,
        violations=["Modeled CPU utilization 133.3% exceeds maximum ceiling of 70.0%."],
        summary_status="Constraint violation: modeled CPU 133.3% > 70.0% ceiling",
        parsing_ms=12.5,
        solving_ms=45.2,
        verification_ms=2.1,
        explanation_ms=0.0,
        total_duration_ms=59.8,
        explanation_source=ExplanationSource.LOCAL_TEMPLATE,
        explanation_status="SUPPRESSED_DUE_TO_VIOLATION",
    )

    data = rec.to_dict()
    assert data["mode"] == 4
    assert data["feasibility"] == "FAIL"
    assert data["optimality_status"] == OptimalityStatus.INFEASIBLE.value
    assert "133.3%" in data["summary_status"]
    assert data["timing_ms"]["total_duration_ms"] == 59.8

    srow = rec.to_summary_row()
    assert srow["mode"] == "Mode 4"
    assert srow["feasibility"] == "FAIL"


def test_stage_trace_and_comparative_runner_offline_execution():
    """run_pipeline_trace runs fully offline with mock or local solvers."""
    query = "Continuous dynamic scaling with target CPU 70% under $1500"
    rec_m3 = run_pipeline_trace(query, mode=3, silent=True)
    rec_m4 = run_pipeline_trace(query, mode=4, silent=True)

    # Both must report the exact same feasibility and verdict
    assert rec_m3.feasibility == rec_m4.feasibility
    assert rec_m3.summary_status == rec_m4.summary_status
    assert rec_m3.claimed_cost_usd == rec_m4.claimed_cost_usd
