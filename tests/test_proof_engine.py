"""Unit tests for the Deterministic Mathematical Proof and Verification Engine."""

import pytest
from src.semantic.schemas import CloudOptimizationContract
from src.verifiers.proof_engine import (
    MathematicalProofEngine,
    compute_optimality_certificate,
    verify_feasibility,
    MATHEMATICAL_FORMULAS,
)
from src.symbolic.models import OptimizationResult, OptimizationMetrics, OptimizationCandidate, FeasibilityStatus


def test_verify_feasibility_perfect_plan():
    """Test feasibility verification on a strictly valid deployment candidate."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=500.0,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.9,
    )

    candidate = {
        "allocated_vcpu": 4,
        "allocated_ram": 16.0,
        "catalog_cost": 320.0,
        "predicted_cost": 320.0,
        "achieved_sla": 99.95,
        "achieved_latency_ms": 45.0,
    }

    res = verify_feasibility(candidate, contract)

    assert res["is_feasible"] is True
    assert res["violation_rate_pct"] == 0.0
    assert res["budget_overflow_usd"] == 0.0
    assert res["cost_error_pct"] == 0.0
    assert res["vcpu_deficit"] == 0
    assert res["ram_deficit"] == 0.0
    assert res["sla_deficit"] == 0.0


def test_verify_feasibility_numerical_deficits():
    """Test explicit numerical constraint deficits calculation when constraints are violated."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=400.0,
        required_vcpus=8,
        required_ram_gb=32.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.99,
    )

    candidate = {
        "allocated_vcpu": 4,        # Deficit: 8 - 4 = 4
        "allocated_ram": 16.0,      # Deficit: 32 - 16 = 16.0
        "catalog_cost": 450.0,      # Overflow: 450 - 400 = 50.0
        "predicted_cost": 300.0,    # Error: |300 - 450| / 450 * 100% = 33.33%
        "achieved_sla": 99.90,      # Deficit: 99.99 - 99.90 = 0.09
        "achieved_latency_ms": 80.0, # Overflow: 80 - 50 = 30.0
    }

    res = verify_feasibility(candidate, contract)

    assert res["is_feasible"] is False
    assert res["vcpu_deficit"] == 4
    assert res["ram_deficit"] == 16.0
    assert res["budget_overflow_usd"] == 50.0
    assert pytest.approx(res["sla_deficit"], 0.001) == 0.09
    assert res["latency_overflow_ms"] == 30.0
    assert res["violation_rate_pct"] == 100.0
    assert pytest.approx(res["cost_error_pct"], 0.1) == 33.33


def test_verify_feasibility_partial_violation():
    """Test partial violation where only budget is exceeded."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=300.0,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.9,
    )

    candidate = {
        "allocated_vcpu": 4,
        "allocated_ram": 16.0,
        "catalog_cost": 350.0,      # Over budget by $50
        "predicted_cost": 350.0,
        "achieved_sla": 99.99,
        "achieved_latency_ms": 20.0,
    }

    res = verify_feasibility(candidate, contract)

    assert res["is_feasible"] is False
    assert res["budget_overflow_usd"] == 50.0
    assert res["vcpu_deficit"] == 0
    assert res["ram_deficit"] == 0.0
    # 1 out of 4 evaluated core constraints violated = 25.0%
    assert res["violation_rate_pct"] == 25.0
    assert res["cost_error_pct"] == 0.0


def test_verify_feasibility_with_allocated_vms_aggregation():
    """Test automatic summation from allocated_vms list."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=500.0,
        required_vcpus=8,
        required_ram_gb=32.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.9,
    )

    candidate = {
        "allocated_vms": [
            {"vcpus_per_vm": 4, "ram_gb_per_vm": 16.0, "count": 2, "monthly_cost": 280.0}
        ],
        "predicted_cost": 280.0,
        "achieved_sla": 99.99,
    }

    res = verify_feasibility(candidate, contract)

    assert res["is_feasible"] is True
    assert res["allocated_vcpu"] == 8
    assert res["allocated_ram"] == 32.0
    assert res["catalog_cost"] == 280.0
    assert res["budget_overflow_usd"] == 0.0


def test_compute_optimality_certificate_highs():
    """Test MIP optimality certificate extraction for SciPy HiGHS."""
    highs_result = {
        "solver_name": "SciPy_HiGHS_MILP",
        "status": "OPTIMAL",
        "is_feasible": True,
        "mip_gap": 0.00005,
        "total_monthly_cost_usd": 240.0,
    }

    cert = compute_optimality_certificate(highs_result)

    assert cert["is_provably_optimal"] is True
    assert cert["mip_gap_pct"] == 0.005
    assert cert["solver_name"] == "SciPy_HiGHS_MILP"


def test_compute_optimality_certificate_z3_sat():
    """Test optimality certificate for Z3 SMT solver exact satisfaction."""
    z3_result = {
        "solver_name": "GraphSteeredZ3",
        "status": "FEASIBLE",
        "is_feasible": True,
        "total_monthly_cost_usd": 310.0,
    }

    cert = compute_optimality_certificate(z3_result)

    assert cert["is_provably_optimal"] is True
    assert cert["mip_gap_pct"] == 0.0
    assert "Z3 Exact" in cert["status"]


def test_compute_optimality_certificate_z3_unsat():
    """Test optimality certificate for Z3 SMT solver infeasible outcome."""
    z3_result = {
        "solver_name": "GraphSteeredZ3",
        "status": "INFEASIBLE",
        "is_feasible": False,
        "error_message": "UNSAT: Budget cap breached",
    }

    cert = compute_optimality_certificate(z3_result)

    assert cert["is_provably_optimal"] is False
    assert cert["mip_gap_pct"] == 100.0


def test_mathematical_proof_engine_full_certificate():
    """Test high-level wrapper MathematicalProofEngine."""
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        budget_max_usd=600.0,
        required_vcpus=4,
        required_ram_gb=16.0,
    )
    candidate = {
        "allocated_vcpu": 4,
        "allocated_ram": 16.0,
        "catalog_cost": 300.0,
        "predicted_cost": 300.0,
        "achieved_sla": 99.95,
        "solver_name": "GraphSteeredZ3",
        "is_feasible": True,
    }

    full_cert = MathematicalProofEngine.generate_full_certificate(candidate, contract)

    assert full_cert["is_valid"] is True
    assert full_cert["feasibility"]["is_feasible"] is True
    assert full_cert["optimality"]["is_provably_optimal"] is True
    assert "vcpu_deficit" in full_cert["formulas"]
