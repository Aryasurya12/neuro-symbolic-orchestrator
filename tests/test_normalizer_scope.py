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


def test_mode1_dr_under_250_parsing():
    """Mode 1 DR-under-$250 text: must parse us-east-1 + us-central1 and a $243 cost."""
    text = """
### Recommended Disaster Recovery Plan

To meet your high availability requirement (99.99% SLA) and latency budget (< 50ms), we propose a multi-region deployment across AWS and GCP:

- **Primary Region**: AWS `us-east-1` ($120.00/month base compute)
- **Secondary Region**: GCP `us-central1` ($115.00/month base standby)
- **Replication / Latency Surcharge**: $8.00/month

**Total Monthly Cost**: $243.00/month

This deployment satisfies the disaster recovery objective and stays strictly under the $250.00/month budget constraint.
"""
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=text,
        problem_type="Z3_Graph_Disaster_Recovery",
        contract_data={"problem_type": "Z3_Graph_Disaster_Recovery", "budget_max_usd": 250.0},
    )

    assert status == NormalizationStatus.SUCCESS
    assert cost == 243.00, f"Expected cost $243.00, got {cost}"
    assert decision is not None
    assert decision.get("primary_region") == "us-east-1"
    assert decision.get("secondary_region") == "us-central1"


def test_mode1_refusal_with_alternative_cost():
    """The $122 refusal that mentions "$224" as an alternative: must not read $224 as the plan cost."""
    refusal_text = """
I cannot provide a feasible deployment for this disaster recovery topology within your $122.00 monthly budget. 

The minimum required infrastructure across two regions exceeds the budget:
- Primary compute in us-east-1 requires at least $120.00/mo.
- Standby compute in us-central1 requires at least $96.00/mo.
- Inter-region data replication adds $8.00/mo.

A minimal valid deployment would cost $224.00 per month, which exceeds your $122 budget limit.
Therefore, no feasible configuration is possible under the specified financial constraint.
"""
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=refusal_text,
        problem_type="Z3_Graph_Disaster_Recovery",
        contract_data={"problem_type": "Z3_Graph_Disaster_Recovery", "budget_max_usd": 122.0},
    )

    assert status == NormalizationStatus.SOLVER_INFEASIBLE
    assert cost is None, f"Expected cost None for refusal, but got {cost}"
    assert decision is None


def test_mode1_unicode_non_breaking_hyphens_in_region_names():
    """Text containing non-breaking hyphens (U+2011), en-dash (U+2013), and non-breaking spaces (U+00A0) in region names."""
    # Using \u2011 (non-breaking hyphen), \u2013 (en-dash), \u00a0 (non-breaking space)
    text = (
        "Recommended Disaster Recovery Pair:\n"
        "Primary Region:\u00a0us\u2011east\u20111 (AWS)\n"
        "Secondary Region:\u00a0us\u2013central1 (GCP)\n"
        "Total Monthly Cost:\u00a0$243.00/month"
    )
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=text,
        problem_type="Z3_Graph_Disaster_Recovery",
        contract_data={"problem_type": "Z3_Graph_Disaster_Recovery", "budget_max_usd": 250.0},
    )

    assert status == NormalizationStatus.SUCCESS
    assert cost == 243.00
    assert decision is not None
    assert decision.get("primary_region") == "us-east-1"
    assert decision.get("secondary_region") == "us-central1"


def test_mode1_markdown_total_cost_extraction():
    """Test extracting cost from markdown bold formatting: '**Total monthly cost:** **$243.00**' -> 243.00."""
    text = (
        "Here is the recommended disaster recovery topology:\n"
        "- Primary: AWS us-east-1\n"
        "- Secondary: GCP us-central1\n"
        "**Total monthly cost:** **$243.00**\n"
    )
    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=text,
        problem_type="Z3_Graph_Disaster_Recovery",
        contract_data={"problem_type": "Z3_Graph_Disaster_Recovery", "budget_max_usd": 250.0},
    )

    assert status == NormalizationStatus.SUCCESS
    assert cost == 243.00, f"Expected 243.00, got {cost}"


def test_mode1_unit_price_arithmetic_excluded():
    """Test that unit prices preceded by '×' or followed by '/ms' ('32 ms × $0.25 = $8') are never used as plan cost."""
    text = (
        "Disaster Recovery proposal:\n"
        "- Primary region: AWS us-east-1 ($120.00/mo)\n"
        "- Secondary region: GCP us-central1 ($115.00/mo)\n"
        "Inter-region sync latency calculation: 32 ms × $0.25 = $8\n"
        "**Total monthly cost:** **$243.00**\n"
    )
    # Check candidate filter strictly excludes 0.25
    candidates = OutputNormalizer._filter_monthly_dollar_candidates(text)
    candidate_vals = [c[0] for c in candidates]
    assert 0.25 not in candidate_vals, f"Unit price 0.25 must be excluded from candidates: {candidate_vals}"

    status, cost, decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=text,
        problem_type="Z3_Graph_Disaster_Recovery",
        contract_data={"problem_type": "Z3_Graph_Disaster_Recovery", "budget_max_usd": 250.0},
    )
    assert status == NormalizationStatus.SUCCESS
    assert cost == 243.00, f"Expected total cost $243.00, not unit price $0.25: got {cost}"
    assert decision.get("primary_region") == "us-east-1"
    assert decision.get("secondary_region") == "us-central1"


def test_independent_checker_violations_enforce_no_pass():
    """If any violation is listed, the verdict and Alloc Feas must not say Feasible/PASS."""
    contract_data = {
        "problem_type": "Z3_Graph_Disaster_Recovery",
        "budget_max_usd": 200.0,  # Lower than recomputed $243
        "sla_availability_pct": 99.99,
        "latency_max_ms": 50.0,
        "cloud_providers": ["AWS", "GCP"],
    }
    solver_res = {
        "primary_region": "us-east-1",
        "secondary_region": "us-central1",
        "total_monthly_cost_usd": 243.00,
        "solver": "Raw_LLM_Prose",
    }

    check = IndependentChecker.verify_disaster_recovery(contract_data, solver_res)
    assert len(check["violations"]) > 0, "Budget overflow must produce violations"
    assert check["feasible_against_contract"] is False, "Feasible must be False when violations exist"
    assert "Feasible" not in check["summary_status"], f"Summary status must not say Feasible: {check['summary_status']}"


def test_independent_checker_reports_plan_validity_and_cost_accuracy_separately():
    """Plan validity (feasible_against_contract) and claimed-cost accuracy (cost_accuracy) are reported as separate fields."""
    contract_data = {
        "problem_type": "ILP_VM_Allocation",
        "required_vcpus": 4,
        "required_ram_gb": 8.0,
        "budget_max_usd": 500.0,
        "cloud_providers": ["AWS"],
    }
    # 1x t3.large = 2 vCPUs, 8GB RAM (vCPU deficit -> invalid plan)
    solver_res = {
        "allocated_vms": [{"sku": "t3.large", "provider": "AWS", "count": 1}],
        "total_monthly_cost_usd": 60.74,
        "solver": "Mode1_Prose_LLM",
    }
    check = IndependentChecker.verify_vm_allocation(contract_data, solver_res)
    assert "cost_accuracy" in check
    assert "reported_cost_usd" in check["cost_accuracy"]
    assert "calculated_catalog_cost_usd" in check["cost_accuracy"]
    assert "feasible_against_contract" in check
    assert check["feasible_against_contract"] is False


def test_mode1_and_mode2_optimality_labels_not_proven():
    """For Modes 1 and 2, verify solver does not receive 'Exact Discrete Minimum' or 'Provably Optimal'."""
    contract_data = {
        "problem_type": "Z3_Graph_Disaster_Recovery",
        "budget_max_usd": 300.0,
        "sla_availability_pct": 99.99,
        "latency_max_ms": 50.0,
        "cloud_providers": ["AWS", "GCP"],
    }
    solver_res_m1 = {
        "primary_region": "us-east-1",
        "secondary_region": "us-central1",
        "total_monthly_cost_usd": 243.00,
        "solver": "Raw_LLM_Prose",
    }
    check_m1 = IndependentChecker.verify_disaster_recovery(contract_data, solver_res_m1)
    assert check_m1["feasible_against_contract"] is True
    assert check_m1["optimality_verdict"] == "Matches independent optimum (not proven by this mode)"
    assert "Exact Discrete Minimum" not in check_m1["optimality_verdict"]
    assert "Provably Optimal" not in check_m1["optimality_verdict"]


