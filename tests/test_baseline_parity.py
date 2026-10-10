"""Automated Test Suite for Experimental Fairness and Baseline Parity.

Verifies:
1. Mode 2 Disaster Recovery and Continuous Scaling schemas are evaluated fairly without VM-only bias.
2. Mode 1 natural language prose proofs of infeasibility are recognized.
3. Mode 3 constraint verification rejects misparsed Hindi numbers (no false positives).
4. Information parity: Catalog snapshots in LLM prompts contain all necessary domain facts.
5. Solver isolation: LLM baselines do not secretly call deterministic solvers.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.deterministic_scorer import DeterministicScorer, IndependentCatalogOracle
from src.semantic.normalizer import OutputNormalizer
from src.verifiers.canonical_record import NormalizationStatus


@pytest.fixture
def scorer():
    oracle = IndependentCatalogOracle()
    return DeterministicScorer(oracle=oracle)


# 1. Mode 2 Disaster Recovery Schema Parity
def test_mode2_dr_schema_parity(scorer):
    """Verifies that a valid DR JSON schema is evaluated as a verified success, not TASK_INCOMPATIBLE."""
    manifest_dr = {
        "query_id": "Q11_DR_AWS_GCP_50MS_600USD_FEASIBLE",
        "intended_archetype": "Z3_Graph_Disaster_Recovery",
        "expected_outcome": "FEASIBLE",
        "sla_availability_pct": 99.99,
        "latency_max_ms": 50.0,
        "budget_max_usd": 600.0,
        "optimal_cost_usd": 243.0,
    }

    raw_mode2_dr_json = {
        "task_type": "Z3_Graph_Disaster_Recovery",
        "outcome": "ready",
        "primary_region": "us-east-1",
        "secondary_region": "us-central1",
        "total_monthly_cost_usd": 243.0,
    }

    verdict = scorer.evaluate_record(
        query_id="Q11_DR_AWS_GCP_50MS_600USD_FEASIBLE",
        mode=2,
        manifest_entry=manifest_dr,
        raw_output=raw_mode2_dr_json,
    )

    assert verdict.strict_success == 1
    assert verdict.constraints_satisfied is True
    assert verdict.cost_correct is True
    assert verdict.outcome_correct is True
    assert verdict.explain_label == "CORRECT"


# 2. Mode 2 Continuous Scaling Schema Parity
def test_mode2_scaling_schema_parity(scorer):
    """Verifies that a valid Continuous Scaling JSON schema is evaluated as a verified success."""
    manifest_scaling = {
        "query_id": "Q17_SCALING_300MBPS_60CPU_FEASIBLE",
        "intended_archetype": "PSO_Continuous_Scaling",
        "expected_outcome": "FEASIBLE",
        "target_bandwidth_mbps": 300.0,
        "max_cpu_pct": 60.0,
        "budget_max_usd": 1500.0,
        "optimal_cost_usd": 339.0,
    }

    raw_mode2_scaling_json = {
        "task_type": "PSO_Continuous_Scaling",
        "outcome": "ready",
        "optimal_bandwidth_mbps": 300.0,
        "recommended_replicas": 7,
        "total_monthly_cost_usd": 339.0,
    }

    verdict = scorer.evaluate_record(
        query_id="Q17_SCALING_300MBPS_60CPU_FEASIBLE",
        mode=2,
        manifest_entry=manifest_scaling,
        raw_output=raw_mode2_scaling_json,
    )

    assert verdict.strict_success == 1
    assert verdict.constraints_satisfied is True
    assert verdict.cost_correct is True
    assert verdict.outcome_correct is True
    assert verdict.explain_label == "CORRECT"


# 3. Mode 1 Natural Language Infeasibility Proof Fairness
def test_mode1_prose_infeasibility_fairness(scorer):
    """Verifies that an unstructured natural language proof of infeasibility is captured and queued."""
    manifest_infeasible = {
        "query_id": "Q04_VM_AWS_16VCPU_64GB_INFEASIBLE",
        "intended_archetype": "ILP_VM_Allocation",
        "expected_outcome": "INFEASIBLE",
        "required_vcpus": 16,
        "required_ram_gb": 64.0,
        "budget_max_usd": 450.0,
    }

    raw_mode1_prose = (
        "With the current AWS SKU pricing in the catalog, a deployment that provides "
        "16 vCPUs + 64 GB RAM cannot be built for <= $450 / month. "
        "The cheapest feasible combination is 2 x m5.2xlarge costing $560.64 / month, "
        "which exceeds your budget by $110.64."
    )

    verdict = scorer.evaluate_record(
        query_id="Q04_VM_AWS_16VCPU_64GB_INFEASIBLE",
        mode=1,
        manifest_entry=manifest_infeasible,
        raw_output=raw_mode1_prose,
    )

    assert verdict.strict_success == 1
    assert verdict.outcome_correct is True
    assert verdict.explain_label == "CORRECT_REFUSAL"


# 4. Mode 3 Pure Symbolic Rejection on Misparsed Hindi Numbers
def test_mode3_hindi_words_rejection(scorer):
    """Verifies that when Pure Symbolic regex misparses Hindi numbers, it fails constraints rather than receiving a false-positive pass."""
    manifest = {
        "query_id": "Q03_VM_HINGLISH_WORDS_8VCPU_16GB",
        "intended_archetype": "ILP_VM_Allocation",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 300.0,
        "cloud_providers": ["AWS"],
    }

    # Pure symbolic defaulted to 1 vCPU / 1 GB and provisioned only 1x t3.medium
    misparsed_plan = {
        "allocated_vms": [{"sku": "t3.medium", "quantity": 1}],
        "total_monthly_cost_usd": 30.37,
    }

    contract_misparsed = {
        "required_vcpus": 1,
        "required_ram_gb": 1.0,
        "budget_max_usd": 500.0,
    }

    verdict = scorer.evaluate_record(
        query_id="Q03_VM_HINGLISH_WORDS_8VCPU_16GB",
        mode=3,
        manifest_entry=manifest,
        raw_output=misparsed_plan,
        extracted_contract=contract_misparsed,
    )

    assert verdict.strict_success == 0
    assert verdict.interpretation_correct is False
    assert verdict.constraints_satisfied is False
    assert "Insufficient vCPUs" in verdict.failure_reason


# 5. Information Parity: Catalog Completeness in Prompts
def test_information_parity_catalog_completeness():
    """Verifies that the prompt catalog context in llm_client.py contains all SKUs, regions, and formulas."""
    with open(os.path.join(BASE_DIR, "src", "semantic", "llm_client.py"), "r", encoding="utf-8") as f:
        llm_code = f.read()

    # Check that all 10 VM SKUs are documented in the prompt context
    expected_skus = [
        "t3.medium", "t3.large", "t3.xlarge", "c5.large", "c5.xlarge",
        "m5.large", "m5.xlarge", "m5.2xlarge", "Standard_D4s_v5", "e2-standard-4"
    ]
    for sku in expected_skus:
        assert sku in llm_code, f"Missing SKU '{sku}' in llm_client.py prompt catalog context"

    # Check regions
    expected_regions = ["us-east-1", "us-west-2", "eu-west-1", "eastus", "us-central1"]
    for reg in expected_regions:
        assert reg in llm_code, f"Missing region '{reg}' in llm_client.py prompt catalog context"

    # Check formulas
    assert "75.0 Mbps per replica" in llm_code
    assert "$0.25" in llm_code
    assert "$45.00" in llm_code


# 6. Solver Isolation: LLM Baselines Never Call Mathematical Solvers
def test_solver_isolation_invariant():
    """Verifies that Mode 1 and Mode 2 dispatch functions do not import or invoke SciPy, Z3, or PSO solvers."""
    with open(os.path.join(BASE_DIR, "src", "semantic", "llm_client.py"), "r", encoding="utf-8") as f:
        llm_code = f.read()

    assert "solve_ilp_vm_knapsack" not in llm_code
    assert "solve_z3_graph_disaster_recovery" not in llm_code
    assert "solve_pso_continuous_scaling" not in llm_code
    assert "scipy.optimize" not in llm_code
    assert "import z3" not in llm_code
