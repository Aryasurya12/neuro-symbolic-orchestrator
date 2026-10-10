"""Comprehensive Unit Test Suite for Deterministic Evaluation Scorer.

Tests all 15 required evaluation edge cases and scoring dimensions defined in
research_eval/EVALUATION_PROTOCOL.md:
1. Correct feasible optimization
2. Correct infeasibility
3. Unsupported requests
4. Missing critical specifications (clarification required)
5. Contradictory requirements (conflicting)
6. Wrong budget extraction
7. Wrong CPU or RAM extraction
8. Incorrect cloud provider
9. Incorrect SLA or latency
10. Wrong total cost
11. Correct answer obtained from wrong input (RIGHT_ANSWER_WRONG_READING)
12. Invalid or missing output
13. Correct feasible solution without valid optimality claim (suboptimal / heuristic)
14. Solver timeout / provider failure
15. Failed independent verification
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.deterministic_scorer import DeterministicScorer, IndependentCatalogOracle


@pytest.fixture
def scorer():
    oracle = IndependentCatalogOracle()
    return DeterministicScorer(oracle=oracle)


# 1. Correct Feasible Optimization
def test_correct_feasible_optimization(scorer):
    manifest = {
        "query_id": "Q01_VM_AWS_8VCPU_16GB_FEASIBLE",
        "expected_outcome": "FEASIBLE",
        "intended_archetype": "ILP_VM_Allocation",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 300.0,
        "cloud_providers": ["AWS"],
        "optimal_cost_usd": 242.94,
    }
    raw_output = {
        "task_type": "ILP_VM_Allocation",
        "outcome": "ready",
        "allocated_vms": [{"sku": "t3.xlarge", "provider": "AWS", "quantity": 2, "monthly_cost": 242.94}],
        "total_monthly_cost_usd": 242.94,
    }
    contract = {
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 300.0,
        "cloud_providers": ["AWS"],
    }
    verdict = scorer.evaluate_record("Q01", 4, manifest, raw_output, extracted_contract=contract)
    assert verdict.strict_success == 1
    assert verdict.interpretation_correct is True
    assert verdict.outcome_correct is True
    assert verdict.constraints_satisfied is True
    assert verdict.cost_correct is True
    assert verdict.optimality_status == "PROVED_OPTIMAL"
    assert verdict.explain_label == "CORRECT"


# 2. Correct Infeasibility
def test_correct_infeasibility(scorer):
    manifest = {
        "query_id": "Q04_VM_AWS_16VCPU_64GB_INFEASIBLE",
        "expected_outcome": "INFEASIBLE",
        "intended_archetype": "ILP_VM_Allocation",
        "required_vcpus": 16,
        "required_ram_gb": 64.0,
        "budget_max_usd": 450.0,
    }
    raw_output = {
        "task_type": "ILP_VM_Allocation",
        "outcome": "infeasible",
        "reason": "Cost exceeds budget.",
    }
    verdict = scorer.evaluate_record("Q04", 2, manifest, raw_output)
    assert verdict.strict_success == 1
    assert verdict.outcome_correct is True
    assert verdict.refusal_correct is True
    assert verdict.explain_label == "CORRECT_REFUSAL"


# 3. Unsupported Request Rejection
def test_unsupported_request_rejection(scorer):
    manifest = {
        "query_id": "Q23_UNSUPPORTED_BIRYANI_RECIPE",
        "expected_outcome": "UNSUPPORTED",
    }
    raw_output = "I’m sorry, but I can’t help with that recipe."
    verdict = scorer.evaluate_record("Q23", 1, manifest, raw_output)
    assert verdict.strict_success == 1
    assert verdict.outcome_correct is True
    assert verdict.refusal_correct is True
    assert verdict.explain_label == "CORRECT_REFUSAL"


# 4. Missing Critical Specifications (Clarification Required)
def test_missing_critical_specifications(scorer):
    manifest = {
        "query_id": "Q07_VM_AWS_MISSING_BUDGET",
        "expected_outcome": "CLARIFICATION_REQUIRED",
    }
    # Case A: System correctly asked for clarification
    raw_output_good = {
        "task_type": "needs_clarification",
        "outcome": "needs_clarification",
        "reason": "Missing budget.",
    }
    v_good = scorer.evaluate_record("Q07", 2, manifest, raw_output_good)
    assert v_good.strict_success == 1
    assert v_good.explain_label == "CORRECT_REFUSAL"

    # Case B: System guessed a plan
    raw_output_bad = {
        "task_type": "ILP_VM_Allocation",
        "outcome": "ready",
        "allocated_vms": [{"sku": "t3.large", "provider": "AWS", "quantity": 1}],
        "total_monthly_cost_usd": 60.74,
    }
    v_bad = scorer.evaluate_record("Q07", 2, manifest, raw_output_bad)
    assert v_bad.strict_success == 0
    assert v_bad.outcome_correct is False
    assert v_bad.explain_label == "WRONG_OUTCOME"


# 5. Contradictory Requirements (Conflicting)
def test_contradictory_requirements(scorer):
    manifest = {
        "query_id": "Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING",
        "expected_outcome": "CONFLICTING_REQUIREMENTS",
        "target_cpu_pct": 70.0,
        "max_cpu_pct": 60.0,
    }
    telemetry = {
        "proof_status": "Conflicting Requirements",
        "explain_label": "CORRECT_REFUSAL",
        "plan_valid": False,
    }
    verdict = scorer.evaluate_record("Q19", 4, manifest, {}, telemetry_record=telemetry)
    assert verdict.strict_success == 1
    assert verdict.outcome_correct is True
    assert verdict.refusal_correct is True
    assert verdict.explain_label == "CORRECT_REFUSAL"


# 6. Wrong Budget Extraction
def test_wrong_budget_extraction(scorer):
    manifest = {
        "query_id": "Q01_VM_AWS_8VCPU_16GB_FEASIBLE",
        "expected_outcome": "FEASIBLE",
        "budget_max_usd": 300.0,
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
    }
    contract = {
        "budget_max_usd": 150.0,  # Wrong budget extracted!
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
    }
    raw_output = {
        "allocated_vms": [{"sku": "t3.xlarge", "provider": "AWS", "quantity": 2, "monthly_cost": 242.94}],
        "total_monthly_cost_usd": 242.94,
    }
    verdict = scorer.evaluate_record("Q01", 4, manifest, raw_output, extracted_contract=contract)
    assert verdict.interpretation_correct is False
    assert "budget_max_usd" in verdict.mismatch_fields[0]
    assert verdict.strict_success == 0
    assert verdict.explain_label == "RIGHT_ANSWER_WRONG_READING"


# 7. Wrong CPU or RAM Extraction
def test_wrong_cpu_or_ram_extraction(scorer):
    manifest = {
        "query_id": "Q01_VM_AWS_8VCPU_16GB_FEASIBLE",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 300.0,
    }
    contract = {
        "required_vcpus": 4,  # Under-extracted!
        "required_ram_gb": 8.0,
        "budget_max_usd": 300.0,
    }
    raw_output = {
        "allocated_vms": [{"sku": "t3.large", "provider": "AWS", "quantity": 2, "monthly_cost": 60.74}],
        "total_monthly_cost_usd": 60.74,
    }
    verdict = scorer.evaluate_record("Q01", 4, manifest, raw_output, extracted_contract=contract)
    assert verdict.interpretation_correct is False
    assert verdict.strict_success == 0


# 8. Incorrect Cloud Provider
def test_incorrect_cloud_provider(scorer):
    manifest = {
        "query_id": "Q01_VM_AWS_8VCPU_16GB_FEASIBLE",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "cloud_providers": ["AWS"],
        "budget_max_usd": 300.0,
    }
    raw_output = {
        "allocated_vms": [{"sku": "e2-standard-4", "provider": "GCP", "quantity": 2, "monthly_cost": 195.64}],
        "total_monthly_cost_usd": 195.64,
    }
    verdict = scorer.evaluate_record("Q01", 2, manifest, raw_output)
    assert verdict.constraints_satisfied is False
    assert "provider" in verdict.failure_reason.lower()
    assert verdict.strict_success == 0
    assert verdict.explain_label == "WRONG_PLAN"


# 9. Incorrect SLA or Latency (DR)
def test_incorrect_sla_or_latency(scorer):
    manifest = {
        "query_id": "Q11_DR_FEASIBLE",
        "expected_outcome": "FEASIBLE",
        "intended_archetype": "Z3_Graph_Disaster_Recovery",
        "sla_availability_pct": 99.99,
        "latency_max_ms": 20.0,  # Strict latency
        "budget_max_usd": 600.0,
    }
    # Pair us-east-1 and us-central1 has 32ms latency (> 20ms)
    raw_output = {
        "task_type": "Z3_Graph_Disaster_Recovery",
        "outcome": "ready",
        "primary_region": "us-east-1",
        "secondary_region": "us-central1",
        "total_monthly_cost_usd": 243.0,
    }
    verdict = scorer.evaluate_record("Q11", 2, manifest, raw_output)
    assert verdict.constraints_satisfied is False
    assert "latency violation" in verdict.failure_reason.lower()
    assert verdict.strict_success == 0
    assert verdict.explain_label == "WRONG_PLAN"


# 10. Wrong Total Cost (Cost Error)
def test_wrong_total_cost(scorer):
    manifest = {
        "query_id": "Q01_VM_AWS_8VCPU_16GB_FEASIBLE",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 300.0,
        "cloud_providers": ["AWS"],
    }
    # 2x t3.xlarge costs $242.94, but response claimed $50.00
    raw_output = {
        "allocated_vms": [{"sku": "t3.xlarge", "provider": "AWS", "quantity": 2}],
        "total_monthly_cost_usd": 50.0,  # Fabricated low price!
    }
    verdict = scorer.evaluate_record("Q01", 2, manifest, raw_output)
    assert verdict.constraints_satisfied is True
    assert verdict.cost_correct is False
    assert verdict.strict_success == 0


# 11. Right Answer Wrong Reading (Q10 Mode 4 Edge Case)
def test_right_answer_wrong_reading(scorer):
    manifest = {
        "query_id": "Q10_VM_HINGLISH_GCP_AZURE_8VCPU_32GB",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 8,
        "required_ram_gb": 32.0,
        "budget_max_usd": 350.0,
        "cloud_providers": ["GCP", "Azure"],
        "optimal_cost_usd": 293.46,
    }
    # Extracted 48 GB RAM instead of 32 GB requested
    contract = {
        "required_vcpus": 8,
        "required_ram_gb": 48.0,  # Overspecified!
        "budget_max_usd": 350.0,
        "cloud_providers": ["GCP", "Azure"],
    }
    raw_output = {
        "allocated_vms": [{"sku": "e2-standard-4", "provider": "GCP", "quantity": 3}],
        "total_monthly_cost_usd": 293.46,
    }
    verdict = scorer.evaluate_record("Q10", 4, manifest, raw_output, extracted_contract=contract)
    assert verdict.constraints_satisfied is True
    assert verdict.interpretation_correct is False
    assert verdict.strict_success == 0
    assert verdict.explain_label == "RIGHT_ANSWER_WRONG_READING"


# 12. Invalid or Missing Output
def test_invalid_or_missing_output(scorer):
    manifest = {
        "query_id": "Q01_VM_FEASIBLE",
        "expected_outcome": "FEASIBLE",
    }
    verdict = scorer.evaluate_record("Q01", 1, manifest, "Random non-finops gibberish")
    assert verdict.strict_success == 0
    assert verdict.constraints_satisfied is False
    assert verdict.explain_label == "WRONG_OUTCOME"


# 13. Suboptimal Feasible Solution
def test_suboptimal_feasible_solution(scorer):
    manifest = {
        "query_id": "Q01_VM_AWS_8VCPU_16GB_FEASIBLE",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 300.0,
        "cloud_providers": ["AWS"],
        "optimal_cost_usd": 242.94,  # Optimum is 2x t3.xlarge ($242.94)
    }
    # Mode picked 2x m5.xlarge ($280.32) - valid but costlier
    raw_output = {
        "allocated_vms": [{"sku": "m5.xlarge", "provider": "AWS", "quantity": 2}],
        "total_monthly_cost_usd": 280.32,
    }
    verdict = scorer.evaluate_record("Q01", 2, manifest, raw_output)
    assert verdict.constraints_satisfied is True
    assert verdict.optimality_status == "SUBOPTIMAL"
    assert verdict.cost_gap_usd > 0.01
    assert verdict.explain_label == "CORRECT_BUT_COSTLIER"


# 14. Solver Timeout / Provider Failure
def test_solver_timeout(scorer):
    manifest = {
        "query_id": "Q01_VM_FEASIBLE",
        "expected_outcome": "FEASIBLE",
    }
    telemetry = {
        "proof_status": "Timeout",
        "plan_valid": False,
        "explain_label": "PROVIDER_FAILURE",
    }
    verdict = scorer.evaluate_record("Q01", 4, manifest, {}, telemetry_record=telemetry)
    assert verdict.strict_success == 0
    assert verdict.outcome_correct is False
    assert verdict.explain_label == "PROVIDER_FAILURE"


# 15. Failed Independent Verification (Unknown SKU)
def test_failed_independent_verification_unknown_sku(scorer):
    manifest = {
        "query_id": "Q01_VM_FEASIBLE",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
    }
    raw_output = {
        "allocated_vms": [{"sku": "hallucinated.super.instance", "quantity": 1}],
        "total_monthly_cost_usd": 10.0,
    }
    verdict = scorer.evaluate_record("Q01", 2, manifest, raw_output)
    assert verdict.constraints_satisfied is False
    assert "unknown sku" in verdict.failure_reason.lower()
    assert verdict.strict_success == 0
