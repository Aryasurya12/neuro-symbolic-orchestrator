"""Unit tests for the Live Capability Proof module (src/proof/capability_demo.py).

Verifies the formal computational basis for:
1. Mode 3 (Pure Symbolic): Demonstrating 0% NLU capability by passing raw text directly
   into typed numerical solver arguments, capturing real dynamic exceptions.
2. Mode 4 (Full Neuro-Symbolic): Demonstrating 100% mathematical soundness by running
   SCOPE extraction first, then invoking the IDENTICAL underlying solver function.
3. Multi-solver verification across ILP Knapsack, Continuous PSO, and Z3 SMT DR engines.
4. Robustness on colloquial Hinglish and multi-cloud queries.
"""

from __future__ import annotations

import pytest
from src.proof.capability_demo import (
    ProofResult,
    run_mode3_raw_text_proof,
    run_mode4_pipeline_proof,
    generate_comparison_proof,
    run_pso_raw_text_proof,
    run_pso_pipeline_proof,
    generate_pso_comparison_proof,
    run_z3_raw_text_proof,
    run_z3_pipeline_proof,
    generate_z3_comparison_proof,
    generate_solver_proof_by_type,
    generate_comprehensive_capability_proof,
)


def test_mode3_raw_text_proof_ilp_fails_with_captured_exception():
    """Mode 3 must fail with a real Python exception when given raw unparsed query text."""
    query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
    result = run_mode3_raw_text_proof(query)

    assert isinstance(result, ProofResult)
    assert result.mode_name == "Mode 3: Pure Symbolic (ILP Solver)"
    assert result.succeeded is False
    assert result.raw_input_used == query
    assert result.solver_name == "solve_ilp_vm_knapsack"
    assert result.exception_type in ["ValueError", "TypeError"]
    assert result.exception_message is not None and len(result.exception_message) > 0
    assert result.traceback_text is not None and "Traceback" in result.traceback_text
    assert "0% NLU capability" in result.explanation


def test_mode4_pipeline_proof_ilp_succeeds_on_same_solver():
    """Mode 4 must extract parameters via SCOPE and succeed on the IDENTICAL solver function."""
    query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
    result = run_mode4_pipeline_proof(query)

    assert isinstance(result, ProofResult)
    assert result.mode_name == "Mode 4: Full Neuro-Symbolic (ILP Solver)"
    assert result.succeeded is True
    assert result.raw_input_used == query
    assert result.solver_name == "solve_ilp_vm_knapsack"
    assert result.output is not None
    assert result.output["matched_template"] == "ilp_vm_allocation_template.py"
    assert result.output["jaccard_score"] > 0.0

    solver_res = result.output["solver_result"]
    assert solver_res["status"] == "OPTIMAL"
    assert solver_res["total_vcpus"] >= 8
    assert solver_res["total_ram_gb"] >= 16.0
    assert solver_res["total_monthly_cost_usd"] <= 300.0
    assert len(solver_res["allocated_vms"]) > 0
    assert "EXACT SAME solve_ilp_vm_knapsack()" in result.explanation


def test_generate_comparison_proof_ilp_structure():
    """generate_comparison_proof must produce structured comparison dictionary."""
    query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
    proof = generate_comparison_proof(query)

    assert proof["raw_query"] == query
    assert proof["solver_type"] == "ILP_VM_Knapsack"
    assert proof["solver_function"] == "solve_ilp_vm_knapsack"
    assert isinstance(proof["mode3"], ProofResult)
    assert isinstance(proof["mode4"], ProofResult)
    assert proof["mode3"].succeeded is False
    assert proof["mode4"].succeeded is True
    assert "Mode 3 failed with" in proof["summary"]
    assert "Mode 4 succeeded" in proof["summary"]


def test_pso_continuous_scaling_capability_proof():
    """Continuous PSO solver must fail on raw text and succeed on SCOPE extraction."""
    query = "Need real-time streaming autoscale bandwidth from 100 to 800 Mbps with 70% CPU limit on GCP budget $400"
    
    # 1. Mode 3 Raw Text Direct Call
    m3 = run_pso_raw_text_proof(query)
    assert m3.succeeded is False
    assert m3.solver_name == "solve_pso_continuous_scaling"
    assert m3.exception_type in ["TypeError", "ValueError"]
    assert m3.exception_message is not None

    # 2. Mode 4 SCOPE Extraction Call
    m4 = run_pso_pipeline_proof(query)
    assert m4.succeeded is True
    assert m4.solver_name == "solve_pso_continuous_scaling"
    assert m4.output is not None
    assert m4.output["matched_template"] == "pso_continuous_scaling_template.py"
    
    solver_res = m4.output["solver_result"]
    assert solver_res["status"] == "CONVERGED"
    assert solver_res["estimated_monthly_cost_usd"] <= 400.0
    assert solver_res["optimal_bandwidth_mbps"] >= 100.0

    # 3. Comparison Dict
    pso_comp = generate_pso_comparison_proof(query)
    assert pso_comp["mode3"].succeeded is False
    assert pso_comp["mode4"].succeeded is True


def test_z3_graph_disaster_recovery_capability_proof():
    """Z3 SMT solver must fail on raw text and succeed on SCOPE extraction."""
    query = "AWS active-passive DR across us-east-1 and us-west-2 with 99.99% SLA and $850 budget cap."

    # 1. Mode 3 Raw Text Direct Call
    m3 = run_z3_raw_text_proof(query)
    assert m3.succeeded is False
    assert m3.solver_name == "solve_z3_graph_disaster_recovery"
    assert m3.exception_type in ["TypeError", "ValueError"]
    assert m3.exception_message is not None

    # 2. Mode 4 SCOPE Extraction Call
    m4 = run_z3_pipeline_proof(query)
    assert m4.succeeded is True
    assert m4.solver_name == "solve_z3_graph_disaster_recovery"
    assert m4.output is not None
    assert m4.output["matched_template"] == "z3_graph_disaster_recovery_template.py"

    solver_res = m4.output["solver_result"]
    assert solver_res["status"] == "FEASIBLE"
    assert solver_res["is_feasible"] is True
    assert solver_res["total_monthly_cost_usd"] <= 850.0

    # 3. Comparison Dict
    z3_comp = generate_z3_comparison_proof(query)
    assert z3_comp["mode3"].succeeded is False
    assert z3_comp["mode4"].succeeded is True


def test_hinglish_colloquial_capability_proof():
    """Hinglish input must fail Mode 3 with genuine exception and succeed in Mode 4."""
    query = "Mujhe 16 vCPU aur 32GB RAM chahiye AWS par under 25000 INR monthly"
    proof = generate_comparison_proof(query)

    assert proof["mode3"].succeeded is False
    assert proof["mode3"].exception_type in ["ValueError", "TypeError"]
    assert proof["mode4"].succeeded is True
    
    contract = proof["mode4"].output["extracted_contract"]
    assert contract["required_vcpus"] == 16
    assert contract["required_ram_gb"] == 32.0
    assert contract["cloud_providers"] == ["AWS"]
    # 25000 INR / 85 = ~294 USD
    assert 280.0 <= contract["budget_max_usd"] <= 310.0


def test_generate_solver_proof_by_type_dispatch():
    """generate_solver_proof_by_type must correctly dispatch to the requested solver archetype."""
    q = "Sample workload query"
    p_ilp = generate_solver_proof_by_type(q, "ilp")
    assert p_ilp["solver_function"] == "solve_ilp_vm_knapsack"

    p_pso = generate_solver_proof_by_type(q, "pso")
    assert p_pso["solver_function"] == "solve_pso_continuous_scaling"

    p_z3 = generate_solver_proof_by_type(q, "z3")
    assert p_z3["solver_function"] == "solve_z3_graph_disaster_recovery"


def test_generate_comprehensive_capability_proof_all_solvers():
    """generate_comprehensive_capability_proof must return live proofs across all 3 solvers."""
    query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
    comp = generate_comprehensive_capability_proof(query)

    assert "raw_query" in comp
    assert "extracted_contract" in comp
    assert "matched_problem_type" in comp
    assert "active_proof" in comp
    assert "all_proofs" in comp
    assert "ILP_VM_Knapsack" in comp["all_proofs"]
    assert "PSO_Continuous_Scaling" in comp["all_proofs"]
    assert "Z3_Graph_Disaster_Recovery" in comp["all_proofs"]

    for solver_key, p_res in comp["all_proofs"].items():
        assert p_res["mode3"].succeeded is False
        assert p_res["mode3"].exception_type is not None
        assert p_res["mode3"].exception_message is not None
