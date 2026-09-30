"""Unit tests for SAGE-GNN (IJCNN 2024) Graph Neural Network Adapter and MaxSMT Z3 Solver."""

import time
import pytest
import z3

from src.optimizers.sage_gnn_adapter import (
    ComponentNode,
    SAGEGNNBenchmarkTopology,
    SageGNNConstraintAdapter,
    VMNode,
)
from src.optimizers.z3_smt_solver import (
    run_all_sage_gnn_benchmarks,
    solve_sage_gnn_placement,
)


def test_sage_gnn_adapter_initialization_and_logits():
    """Validates adapter initialization, logit formatting via softmax, and prediction importing."""
    adapter = SageGNNConstraintAdapter(weight_scale=100)
    assert adapter.weight_scale == 100

    raw_logits = {
        ("web", "vm_1"): 2.5,
        ("web", "vm_2"): 1.0,
        ("db", "vm_1"): 0.5,
        ("db", "vm_2"): 3.0,
    }
    probs = adapter.format_gnn_logits(raw_logits)
    assert ("web", "vm_1") in probs
    assert ("web", "vm_2") in probs
    assert probs[("web", "vm_1")] > probs[("web", "vm_2")]
    assert probs[("db", "vm_2")] > probs[("db", "vm_1")]
    # Sum of probabilities per component should approximately equal 1.0
    assert abs((probs[("web", "vm_1")] + probs[("web", "vm_2")]) - 1.0) < 0.01


def test_sage_gnn_soft_constraints_injection():
    """Validates that inject_soft_constraints converts probabilities to integer MaxSMT weights and calls add_soft."""
    adapter = SageGNNConstraintAdapter(weight_scale=100)
    opt = z3.Optimize()

    var_a = z3.Bool("place_a")
    var_b = z3.Bool("place_b")
    placement_vars = {("comp_a", "vm_1"): var_a, ("comp_b", "vm_2"): var_b}
    gnn_probs = {("comp_a", "vm_1"): 0.85, ("comp_b", "vm_2"): 0.40}

    soft_assertions = adapter.inject_soft_constraints(opt, placement_vars, gnn_probs, weight_scale=100)

    assert len(soft_assertions) == 2
    assert soft_assertions[0] == (var_a, 85)
    assert soft_assertions[1] == (var_b, 40)


def test_sage_gnn_benchmark_topologies_loading():
    """Validates that all 3 standardized SAGE-GNN (IJCNN 2024) topologies are available and structured."""
    dataset = SageGNNConstraintAdapter.load_benchmark_dataset()

    expected_topologies = ["WordPress_MultiTier", "Oryx2_Lambda_Pipeline", "Secure_Web_Container"]
    for top_name in expected_topologies:
        assert top_name in dataset
        top = dataset[top_name]
        assert isinstance(top, SAGEGNNBenchmarkTopology)
        assert len(top.components) >= 3
        assert len(top.vm_catalog) >= 5
        assert top.budget_max_usd > 0
        assert len(top.gnn_probabilities) > 0


def test_sage_gnn_z3_solver_latency_under_one_second():
    """Verifies that z3.Optimize() successfully solves all SAGE-GNN benchmark topologies in < 1.0 second."""
    dataset = SageGNNConstraintAdapter.load_benchmark_dataset()

    for name, topology in dataset.items():
        start_t = time.perf_counter()
        res = solve_sage_gnn_placement(
            topology=topology,
            weight_scale=100,
            use_gnn_soft_constraints=True,
            timeout_ms=3000,
        )
        elapsed_sec = time.perf_counter() - start_t

        # Critical Performance SLA requirement: Must solve in < 1.0 second
        assert elapsed_sec < 1.0, f"Topology {name} took {elapsed_sec:.3f}s (exceeded 1.0s limit)"
        assert res["runtime_sec"] < 1.0
        assert res["runtime_ms"] < 1000.0

        # Mathematical guarantees
        assert res["is_feasible"] is True
        assert res["status"] == "FEASIBLE"
        assert res["total_monthly_cost_usd"] <= topology.budget_max_usd
        assert res["max_inter_component_latency_ms"] <= topology.max_latency_ms
        assert len(res["component_placements"]) == len(topology.components)
        assert res["gnn_soft_constraints_active"] is True
        assert res["soft_assertions_count"] > 0


def test_secure_web_container_anti_affinity_enforcement():
    """Verifies that Anti-Affinity hard constraint between Ingress and Auth DB is strictly enforced."""
    top = SageGNNConstraintAdapter.get_topology("Secure_Web_Container")
    assert ("ingress_gateway", "auth_db") in top.anti_affinity_pairs

    res = solve_sage_gnn_placement(top)
    assert res["is_feasible"] is True

    ingress_vm = res["component_placements"]["ingress_gateway"]["vm_id"]
    auth_db_vm = res["component_placements"]["auth_db"]["vm_id"]

    # Ingress and Auth DB MUST NOT be placed on the same VM instance!
    assert ingress_vm != auth_db_vm, f"Anti-affinity violated: Ingress and Auth DB both placed on {ingress_vm}"


def test_infeasible_budget_detection():
    """Verifies that an impossibly low budget cap triggers INFEASIBLE / UNSAT cleanly."""
    top = SageGNNConstraintAdapter.get_topology("WordPress_MultiTier")
    top.budget_max_usd = 10.0  # Impossibly low for a 4-tier deployment

    res = solve_sage_gnn_placement(top)
    assert res["is_feasible"] is False
    assert res["status"] == "INFEASIBLE"
    assert res["constraint_status"]["budget_ok"] is False


def test_run_all_sage_gnn_benchmarks_helper():
    """Validates the run_all_sage_gnn_benchmarks batch execution helper."""
    results = run_all_sage_gnn_benchmarks(weight_scale=100)
    assert len(results) == 3
    for name in ["WordPress_MultiTier", "Oryx2_Lambda_Pipeline", "Secure_Web_Container"]:
        assert name in results
        assert results[name]["is_feasible"] is True
        assert results[name]["runtime_sec"] < 1.0
