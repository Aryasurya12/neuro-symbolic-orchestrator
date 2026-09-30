"""Unit tests for the Raw Symbolic Solver Runner (Mode 3 benchmark evaluation)."""

import pytest
from src.semantic.schemas import CloudOptimizationContract
from src.optimizers.raw_symbolic_runner import (
    run_pure_symbolic_raw,
    run_pure_symbolic_structured,
)


def test_run_pure_symbolic_raw_natural_language_crash():
    """Test that pure symbolic solvers immediately crash on natural language queries without NLU translation."""
    query = "Deploy 4 vCPUs and 16GB RAM on AWS with a budget of $300/month and 99.9% SLA"

    res = run_pure_symbolic_raw(query)

    assert res["status"] == "CRASHED (Parsing Error)"
    assert "ValueError: Symbolic solver requires numeric matrices" in res["error_message"]
    assert res["nlu_success"] is False
    assert res["feasibility"] == "N/A (Solver Execution Failed)"
    assert res["violation_rate"] == "100% (Unparseable Input)"
    assert res["catalog_cost"] == 0.0
    assert res["is_feasible"] is False
    assert res["latency_ms"] >= 0.0
    assert res["raw_input"] == query


def test_run_pure_symbolic_raw_arbitrary_text_crash():
    """Test crash on arbitrary unstructured text strings."""
    query = "I need a fast database server in US-East"

    res = run_pure_symbolic_raw(query)

    assert res["status"] == "CRASHED (Parsing Error)"
    assert res["nlu_success"] is False
    assert res["catalog_cost"] == 0.0


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
