"""Unit tests for the Symbolic Solver Runner with local rule-based parsing (Mode 3 benchmark evaluation)."""

import pytest
from src.semantic.schemas import CloudOptimizationContract
from src.optimizers.raw_symbolic_runner import (
    run_symbolic_rule_based,
    run_pure_symbolic_raw,
    run_pure_symbolic_structured,
)


def test_run_symbolic_rule_based_vm_success():
    """Test that Mode 3 parses VM requirements with local regex and solves with SciPy ILP."""
    query = "Deploy 4 vCPUs and 16GB RAM on AWS with a budget of $500/month and 99.9% SLA"

    res = run_symbolic_rule_based(query)

    assert res["status"] in ["OPTIMAL", "FEASIBLE", "Feasible against checked constraints"]
    assert res["nlu_success"] is True
    assert res["is_feasible"] is True
    assert res["catalog_cost"] > 0.0
    assert res["catalog_cost"] <= 500.0
    assert res["allocated_vcpu"] >= 4
    assert res["allocated_ram"] >= 16.0
    assert "independent_check" in res
    assert res["independent_check"]["feasible_against_contract"] is True


def test_run_symbolic_rule_based_dr_success():
    """Test Mode 3 on disaster recovery query."""
    query = "AWS active-passive DR across us-east-1 and us-west-2 with 99.99% SLA and $850 budget cap"

    res = run_symbolic_rule_based(query)

    assert res["nlu_success"] is True
    assert res["catalog_cost"] > 0.0
    assert "independent_check" in res


def test_run_symbolic_rule_based_scaling_success():
    """Test Mode 3 on continuous scaling query."""
    query = "Scale worker nodes from 1 to 20 replicas under 500ms latency and 80% CPU target"

    res = run_symbolic_rule_based(query)

    assert res["nlu_success"] is True
    assert "independent_check" in res


def test_run_symbolic_rule_based_unparseable_query():
    """Test Mode 3 gracefully reports unparseable for completely unrelated text without crashing."""
    query = "Quantum computing circuit optimization with entangled qubits"

    res = run_symbolic_rule_based(query)

    # Should report failure/unparseable honestly without raising an uncaught exception
    assert res["nlu_success"] is False or res["is_feasible"] is False
    assert "independent_check" in res


def test_run_pure_symbolic_structured_success():
    """Test that structured contracts run smoothly through SciPy MILP."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=500.0,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.9,
    )

    res = run_pure_symbolic_structured(contract)

    assert res["status"] == "OPTIMAL"
    assert res["is_feasible"] is True
    assert res["nlu_success"] is True
    assert res["allocated_vcpu"] >= 4
    assert res["allocated_ram"] >= 16.0
    assert res["catalog_cost"] > 0.0
    assert res["catalog_cost"] <= 500.0
    assert res["is_provably_optimal"] is True
    assert res["violation_rate_pct"] == 0.0
    assert res["budget_overflow_usd"] == 0.0
    assert "feasibility_certificate" in res
    assert "optimality_certificate" in res


def test_run_pure_symbolic_structured_dict_input():
    """Test that structured dictionaries are also supported."""
    contract_dict = {
        "required_vcpus": 2,
        "required_ram_gb": 4.0,
        "budget_max_usd": 200.0,
        "cloud_providers": ["AWS"],
        "sla_availability_pct": 99.9,
    }

    res = run_pure_symbolic_structured(contract_dict)

    assert res["is_feasible"] is True
    assert res["catalog_cost"] <= 200.0
