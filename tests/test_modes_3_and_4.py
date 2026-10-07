"""Unit tests for Mode 3 and Mode 4 execution paths, independent checking, and offline verification."""

import pytest
import time
from src.semantic.schemas import CloudOptimizationContract
from src.verifiers.independent_checker import IndependentChecker
from src.optimizers.raw_symbolic_runner import run_symbolic_rule_based
from app import execute_live_pipeline, build_4way_comparison_data


def test_mode_3_rule_based_vm_execution():
    """Verify Mode 3 execution uses local rule-based parsing and SciPy MILP without LLM calls."""
    query = "Deploy 8 vCPUs and 32GB RAM on AWS with budget of $800/month"
    t0 = time.perf_counter()
    m3_res = run_symbolic_rule_based(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert m3_res["mode"] == "Mode 3: Symbolic + rule-based parsing"
    assert m3_res["is_feasible"] is True
    assert m3_res["allocated_vcpu"] >= 8
    assert m3_res["allocated_ram"] >= 32.0
    assert m3_res["catalog_cost"] > 0
    assert m3_res["catalog_cost"] <= 800.0
    assert elapsed_ms < 2000.0  # Must be fast local execution
    assert "independent_check" in m3_res
    check = m3_res["independent_check"]
    assert check["feasible_against_contract"] is True
    assert check["interpretation_correct"] == "Not independently evaluated"


def test_mode_4_live_pipeline_single_execution():
    """Verify Mode 4 executes end-to-end pipeline exactly once and records timing boundaries."""
    query = "Scale worker nodes from 2 to 15 replicas under 500ms latency and 75% CPU target"

    res = execute_live_pipeline(query)

    # Check timing boundaries are measured separately
    assert "parse_latency_ms" in res
    assert "solve_latency_ms" in res
    assert "explanation_latency_ms" in res
    assert "total_latency_ms" in res
    assert res["parse_latency_ms"] >= 0
    assert res["solve_latency_ms"] >= 0
    assert res["explanation_latency_ms"] >= 0

    # Check independent verifications
    assert "m4_check" in res
    assert "m3_check" in res
    assert res["m4_check"]["structure_valid"] is True

    # Check heuristic solver optimality label
    # Continuous scaling uses PSO (heuristic search)
    opt_verdict = res["m4_check"]["optimality_verdict"]
    assert "Optimality not established" in opt_verdict


def test_independent_checker_honest_verdicts():
    """Verify that IndependentChecker outputs honest verdicts without predetermined claims."""
    # Test valid contract
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        required_vcpus=4,
        required_ram_gb=16.0,
        budget_max_usd=500.0,
    )
    solver_res = {
        "status": "OPTIMAL",
        "total_monthly_cost_usd": 121.48,
        "allocated_vms": [
            {"sku": "t3.xlarge", "provider": "AWS", "quantity": 1, "monthly_cost": 121.48, "vcpus": 4, "ram_gb": 16.0}
        ],
    }

    check = IndependentChecker.verify_solution(contract, solver_res)
    assert check["structure_valid"] is True
    assert check["catalog_consistent"] is True
    assert check["feasible_against_contract"] is True
    assert check["summary_status"] == "Feasible against checked constraints"
    assert check["interpretation_correct"] == "Not independently evaluated"


def test_4way_comparison_data_offline_modes():
    """Verify build_4way_comparison_data populates honest Mode 3 and Mode 4 without LLM calls."""
    query = "AWS active-passive DR across us-east-1 and us-west-2 with 99.99% SLA and $850 budget"
    contract = CloudOptimizationContract(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS"],
        budget_max_usd=850.0,
        sla_availability_pct=99.99,
        latency_max_ms=100.0,
    )
    solver_res = {
        "status": "OPTIMAL",
        "total_monthly_cost_usd": 266.25,
        "allocated_vms": [
            {"sku": "t3.xlarge", "provider": "AWS", "quantity": 1, "monthly_cost": 121.48, "vcpus": 4, "ram_gb": 16.0},
            {"sku": "t3.xlarge", "provider": "AWS", "quantity": 1, "monthly_cost": 121.48, "vcpus": 4, "ram_gb": 16.0},
        ],
        "primary_region": "us-east-1",
        "secondary_region": "us-west-2",
        "achieved_sla_pct": 99.99,
        "inter_region_latency_ms": 70.0,
    }

    comp_data = build_4way_comparison_data(
        user_query=query,
        contract=contract,
        solver_res=solver_res,
        solve_latency_ms=12.5,
        total_latency_ms=25.0,
        m1_run=None,
        m2_run=None,
    )

    bench = comp_data["bench_data"]
    assert len(bench) == 4

    # Mode 1 & Mode 2 should be 'not_run'
    assert bench[0]["source"] == "not_run"
    assert bench[1]["source"] == "not_run"

    # Mode 3 & Mode 4 should be populated with honest statuses
    assert "Mode 3" in bench[2]["mode"]
    assert bench[2]["is_feasible"] is True
    assert "Feasible" in bench[2]["math"]

    assert "Mode 4" in bench[3]["mode"]
    assert bench[3]["is_feasible"] is True
    assert "Feasible" in bench[3]["math"]
