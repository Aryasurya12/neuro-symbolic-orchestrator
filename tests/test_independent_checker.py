"""Independent Checker Unit Tests for Frozen Development Fixtures.

Verifies the IndependentChecker ground-truth validation engine against:
1. Feasible VM request (Real solver output & ground truth catalog check).
2. Impossible VM budget (Check that solver infeasibility or budget overflow is reported accurately).
3. Valid cross-provider DR request (Check region IDs, latency, and SLA verification).
4. DR allocation violating required provider separation (Deliberately invalid fixture with 2 AWS regions when AWS+GCP required).
5. Scaling allocation violating CPU ceiling (Deliberately overloaded unclipped CPU fixture).
6. Unknown SKU allocation (Deliberately invalid fixture with invalid/non-catalog SKU).
7. Missing or ambiguous requirements query.
8. Out-of-archetype query.
"""

import pytest
from src.semantic.schemas import CloudOptimizationContract
from src.verifiers.independent_checker import IndependentChecker
from src.optimizers.raw_symbolic_runner import run_symbolic_rule_based


# =============================================================================
# FROZEN DEVELOPMENT FIXTURES (8 Scenarios)
# =============================================================================

# Fixture 1: Feasible VM Request
FIXTURE_1_QUERY = "Deploy 4 vCPUs and 16GB RAM on AWS with a budget of $500/month and 99.9% SLA"
FIXTURE_1_EXPECTED_CONTRACT = {
    "problem_type": "ILP_VM_Allocation",
    "cloud_providers": ["AWS"],
    "required_vcpus": 4,
    "required_ram_gb": 16.0,
    "budget_max_usd": 500.0,
}

# Fixture 2: Impossible VM Budget
FIXTURE_2_QUERY = "Deploy 32 vCPUs and 64GB RAM on AWS with a budget of $10/month"
FIXTURE_2_EXPECTED_CONTRACT = {
    "problem_type": "ILP_VM_Allocation",
    "cloud_providers": ["AWS"],
    "required_vcpus": 32,
    "required_ram_gb": 64.0,
    "budget_max_usd": 10.0,
}

# Fixture 3: Valid Cross-Provider DR Request
FIXTURE_3_QUERY = "Cross-cloud DR between AWS us-east-1 and GCP us-central1 with 99.99% SLA and $900 budget"
FIXTURE_3_EXPECTED_CONTRACT = {
    "problem_type": "Z3_Graph_Disaster_Recovery",
    "cloud_providers": ["AWS", "GCP"],
    "sla_availability_pct": 99.99,
    "budget_max_usd": 900.0,
}

# Fixture 4: Invalid DR Allocation (Violates Provider Separation)
FIXTURE_4_CONTRACT = CloudOptimizationContract(
    problem_type="Z3_Graph_Disaster_Recovery",
    cloud_providers=["AWS", "GCP"],
    budget_max_usd=800.0,
    sla_availability_pct=99.99,
    latency_max_ms=100.0,
)
FIXTURE_4_INVALID_ALLOCATION = {
    "status": "OPTIMAL",
    "primary_region": "us-east-1",     # AWS
    "secondary_region": "us-west-2",   # AWS - Violates multi-provider requirement!
    "total_monthly_cost_usd": 450.0,
    "achieved_sla_pct": 99.99,
    "inter_region_latency_ms": 70.0,
    "allocated_vms": [
        {"sku": "t3.medium", "provider": "AWS", "quantity": 2, "monthly_cost": 60.74, "vcpus": 2, "ram_gb": 4.0},
        {"sku": "t3.medium", "provider": "AWS", "quantity": 2, "monthly_cost": 60.74, "vcpus": 2, "ram_gb": 4.0},
    ],
}

# Fixture 5: Invalid Scaling Allocation (Violates CPU Ceiling - Unclipped)
FIXTURE_5_CONTRACT = CloudOptimizationContract(
    problem_type="PSO_Continuous_Scaling",
    cloud_providers=["AWS"],
    budget_max_usd=500.0,
    latency_max_ms=500.0,
)
FIXTURE_5_OVERLOADED_ALLOCATION = {
    "status": "OPTIMAL",
    "bandwidth_mbps": 1000.0,
    "worker_replicas": 1,              # 1 replica processing 1000 Mbps -> 1000 / (1 * 75) = 1333% CPU
    "total_monthly_cost_usd": 150.0,
    "achieved_cpu_pct": 80.0,          # Claimed 80%, but modeled is 1333%!
    "allocated_vms": [],
}

# Fixture 6: Unknown SKU Allocation
FIXTURE_6_CONTRACT = CloudOptimizationContract(
    problem_type="ILP_VM_Allocation",
    cloud_providers=["AWS"],
    budget_max_usd=500.0,
    required_vcpus=4,
    required_ram_gb=16.0,
)
FIXTURE_6_UNKNOWN_SKU_ALLOCATION = {
    "status": "OPTIMAL",
    "total_monthly_cost_usd": 100.0,
    "allocated_vms": [
        {"sku": "nonexistent.fake.custom.xlarge", "provider": "AWS", "quantity": 1, "monthly_cost": 100.0, "vcpus": 4, "ram_gb": 16.0}
    ],
}

# Fixture 7: Missing or Ambiguous Requirements
FIXTURE_7_QUERY = "Deploy something fast somewhere"

# Fixture 8: Out-of-Archetype Request
FIXTURE_8_QUERY = "Quantum circuit simulation tensor network optimization"


# =============================================================================
# TEST CASES
# =============================================================================

def test_fixture_1_feasible_vm():
    """Fixture 1: Feasible VM request must pass independent checker with 0 violations."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        required_vcpus=4,
        required_ram_gb=16.0,
        budget_max_usd=500.0,
    )
    # Run Mode 3 solver
    m3_res = run_symbolic_rule_based(FIXTURE_1_QUERY, contract=contract)
    check = IndependentChecker.verify_solution(contract, m3_res)

    assert check["structure_valid"] is True
    assert check["catalog_consistent"] is True
    assert check["feasible_against_contract"] is True
    assert len(check["violations"]) == 0
    assert check["summary_status"] == "Feasible against checked constraints"
    assert check["cost_accuracy"]["calculated_catalog_cost_usd"] > 0
    assert check["cost_accuracy"]["calculated_catalog_cost_usd"] <= 500.0


def test_fixture_2_impossible_vm_budget():
    """Fixture 2: Impossible budget ($10 for 32 vCPU) must report infeasibility / budget violation."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        required_vcpus=32,
        required_ram_gb=64.0,
        budget_max_usd=10.0,
    )
    m3_res = run_symbolic_rule_based(FIXTURE_2_QUERY, contract=contract)
    check = IndependentChecker.verify_solution(contract, m3_res)

    # Either the solver reported infeasible or independent checker caught budget violation
    assert check["feasible_against_contract"] is False
    assert check["summary_status"] in ["Solver reported infeasible", "Constraint violation found"]
    assert len(check["violations"]) > 0 or check["summary_status"] == "Solver reported infeasible"


def test_fixture_3_valid_cross_provider_dr():
    """Fixture 3: Valid cross-provider DR request with valid AWS + GCP regions."""
    contract = CloudOptimizationContract(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "GCP"],
        sla_availability_pct=99.99,
        latency_max_ms=100.0,
        budget_max_usd=900.0,
    )
    m3_res = run_symbolic_rule_based(FIXTURE_3_QUERY, contract=contract)
    check = IndependentChecker.verify_solution(contract, m3_res)

    assert check["structure_valid"] is True
    assert check["catalog_consistent"] is True
    assert check["feasible_against_contract"] is True
    assert check["recomputed_metrics"]["primary_region"] != check["recomputed_metrics"]["secondary_region"]


def test_fixture_4_invalid_provider_separation():
    """Fixture 4: Deliberately invalid DR allocation (both regions AWS when AWS+GCP required)."""
    check = IndependentChecker.verify_solution(FIXTURE_4_CONTRACT, FIXTURE_4_INVALID_ALLOCATION)

    assert check["feasible_against_contract"] is False
    assert any("Multi-cloud request requires distinct providers" in v for v in check["violations"])
    assert check["summary_status"] == "Constraint violation found"


def test_fixture_5_invalid_scaling_overloaded_cpu():
    """Fixture 5: Deliberately overloaded continuous scaling allocation with unclipped CPU."""
    check = IndependentChecker.verify_solution(FIXTURE_5_CONTRACT, FIXTURE_5_OVERLOADED_ALLOCATION)

    assert check["feasible_against_contract"] is False
    assert check["recomputed_metrics"]["modeled_cpu_pct"] > 100.0  # Must NOT be clipped to 100%!
    assert any("Modeled CPU utilization" in v and "exceeds 100%" in v for v in check["violations"])
    assert "Constraint violation" in check["summary_status"]


def test_fixture_6_unknown_sku_rejected():
    """Fixture 6: Unknown SKU must be flagged by catalog consistency checker."""
    check = IndependentChecker.verify_solution(FIXTURE_6_CONTRACT, FIXTURE_6_UNKNOWN_SKU_ALLOCATION)

    assert check["catalog_consistent"] is False
    assert check["feasible_against_contract"] is False
    assert any("Unknown SKU" in v for v in check["violations"])


def test_fixture_7_missing_ambiguous_requirements():
    """Fixture 7: Ambiguous text without explicit parameters."""
    m3_res = run_symbolic_rule_based(FIXTURE_7_QUERY)
    check = m3_res.get("independent_check")

    assert check is not None
    assert check["interpretation_correct"] == "Not independently evaluated"


def test_fixture_8_out_of_archetype():
    """Fixture 8: Out-of-archetype quantum circuit query."""
    m3_res = run_symbolic_rule_based(FIXTURE_8_QUERY)
    check = m3_res.get("independent_check")

    assert check is not None
    # Mode 3 must handle gracefully without crashing
    assert "summary_status" in check
