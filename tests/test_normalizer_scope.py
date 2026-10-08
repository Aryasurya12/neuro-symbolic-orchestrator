"""Regression tests for Mode 1 Scope-Aware Cost Normalization and Semantic Verdicts."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.semantic.normalizer import OutputNormalizer
from src.verifiers.canonical_record import NormalizationStatus, CheckStatus, FeasibilityStatus
from src.verifiers.independent_checker import IndependentChecker


def test_captured_three_tier_response_regression():
    """Test the exact live captured response where VM ($60.74), Scaling ($590.40), and Grand Total ($1039.30) are present.
    
    The normalizer must extract $590.40 for a scaling query, not $60.74.
    """
    captured_text = """
We have prepared a comprehensive multi-tier cloud infrastructure proposal:

1. Base Virtual Machine Allocation:
   - 1x t3.medium instance on AWS: $60.74/month
   - Subtotal for compute nodes: $60.74/month

2. Continuous Dynamic Scaling Workload:
   - Traffic provisioning: 150 Mbps bandwidth ($12.00/month)
   - 4x worker replicas ($180.00/month base + compute)
   - Scaling subtotal: $590.40/month

3. Multi-Region Disaster Recovery:
   - Secondary region in us-west-2: $248.00/month

Overall deployment total across all components: $1,039.30/month
"""
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=captured_text,
        problem_type="PSO_Continuous_Scaling",
        contract_data={"problem_type": "PSO_Continuous_Scaling", "budget_max_usd": 1500.0},
    )

    assert status == NormalizationStatus.SUCCESS
    assert cost == 590.40, f"Expected scaling subtotal $590.40, got {cost}"
    
    # Verify evidence preserved labeled subtotals and unsolicited VM/DR scope expansion
    evidence_fields = {e.field_name: e.extracted_value for e in evidence}
    assert evidence_fields.get("scaling_subtotal_cost_usd") == 590.40
    assert evidence_fields.get("grand_total_monthly_cost_usd") == 1039.30
    assert evidence_fields.get("unsolicited_vm_cost_usd") == 60.74
    assert evidence_fields.get("unsolicited_dr_cost_usd") == 248.00
    assert any("scope expansion" in err.lower() for err in errors)


def test_ordinary_single_total_response():
    """Test ordinary single-total response extracts claimed amount accurately."""
    single_total_text = (
        "To achieve continuous dynamic scaling with target CPU 70%, we recommend provisioning "
        "150 Mbps bandwidth with 4 worker replicas. The estimated total monthly cost is $590.40/month."
    )
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=single_total_text,
        problem_type="PSO_Continuous_Scaling",
        contract_data={"problem_type": "PSO_Continuous_Scaling", "budget_max_usd": 1500.0},
    )

    assert status == NormalizationStatus.SUCCESS
    assert cost == 590.40
    assert (decision.get("optimal_bandwidth_mbps") or decision.get("bandwidth_mbps")) == 150.0
    assert (decision.get("recommended_replicas") or decision.get("replicas")) == 4


def test_ambiguous_unstructured_cost_response():
    """Test that ambiguous responses without clear scoped totals are flagged as AMBIGUOUS/NEEDS_REVIEW."""
    ambiguous_text = (
        "We could maybe look at $45/mo for some things, or perhaps $120/mo, or $300/mo depending on usage."
    )
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=ambiguous_text,
        problem_type="PSO_Continuous_Scaling",
        contract_data={"problem_type": "PSO_Continuous_Scaling", "budget_max_usd": 1500.0},
    )

    assert status in [NormalizationStatus.AMBIGUOUS, NormalizationStatus.NEEDS_REVIEW]


def test_independent_checker_cost_mismatch_semantics():
    """Verifies that a cost claim mismatch does NOT mark a physically and financially valid allocation as infeasible."""
    contract_data = {
        "problem_type": "ILP_VM_Allocation",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 500.0,
        "cloud_providers": ["AWS"],
    }
    # 2x t3.xlarge = 8 vCPUs, 32GB RAM, recomputed cost ~$243.24/mo (<= $500 budget)
    # LLM incorrectly claims $60.00/mo
    solver_res = {
        "allocated_vms": [
            {"sku": "t3.xlarge", "provider": "AWS", "count": 2}
        ],
        "total_monthly_cost_usd": 60.00,
        "solver": "Mode1_Prose_LLM",
        "status": "FEASIBLE",
    }

    check = IndependentChecker.verify_vm_allocation(contract_data, solver_res)

    # Physical constraints and actual cost <= budget are satisfied
    assert check["feasible_against_contract"] is True, "Valid allocation must remain feasible despite cost claim mismatch"
    # Cost accuracy captures the delta
    assert check["cost_accuracy"]["cost_delta_usd"] > 100.0
    pricing_check = next((c for c in check["parameter_checks"] if c["check_id"] == "CHK_VM_PRICING"), None)
    assert pricing_check is not None
    assert pricing_check["status"] == CheckStatus.FAIL.value


def test_truncation_failure_mode1_normalizer():
    """Verify that empty/truncated responses with finish_reason='length' produce TRUNCATION_FAILURE."""
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text="",
        problem_type="ILP_VM_Allocation",
        contract_data={"problem_type": "ILP_VM_Allocation", "budget_max_usd": 300.0},
        finish_reason="length",
    )
    assert status == NormalizationStatus.TRUNCATION_FAILURE
    assert any("Provider/config truncation failure" in err for err in errors)


def test_truncation_failure_mode2_normalizer():
    """Verify that Mode 2 normalizer classifies finish_reason='length' as TRUNCATION_FAILURE."""
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode2_json(
        raw_json_or_text="",
        requested_problem_type="ILP_VM_Allocation",
        contract_data={"problem_type": "ILP_VM_Allocation", "budget_max_usd": 300.0},
        finish_reason="length",
    )
    assert status == NormalizationStatus.TRUNCATION_FAILURE
    assert any("Provider/config truncation failure" in err for err in errors)


def test_optimality_status_verdict_distinctions():
    """Verify OptimalityStatus supports specific non-optimal verdicts."""
    from src.verifiers.canonical_record import OptimalityStatus
    assert OptimalityStatus.TRUNCATION_FAILURE == OptimalityStatus("Provider Truncation (Max Tokens)")
    assert OptimalityStatus.CLARIFICATION_REQUIRED == OptimalityStatus("Clarification Required")
    assert OptimalityStatus.UNSUPPORTED == OptimalityStatus("Unsupported Workload")
    assert OptimalityStatus.CONFLICTING == OptimalityStatus("Conflicting Requirements")
    assert OptimalityStatus.INFRASTRUCTURE_FAILURE == OptimalityStatus("Infrastructure / API Failure")
    assert OptimalityStatus.INFEASIBLE == OptimalityStatus("Infeasible")
