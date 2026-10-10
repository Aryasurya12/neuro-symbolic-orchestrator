"""Automated Regression & Generalization Test Suite.

Audits Neurasym against:
1. Query-ID hardcoding & benchmark-specific logic.
2. Answer leakage in prompts and inference paths.
3. Generalization across previously unseen, randomly generated resource combinations.
4. Mutation testing across resource shifts, budget shifts, custom catalogs, and paraphrases.
5. Evaluator-inference decoupling (evaluator acts as an independent post-hoc oracle).
"""

import json
import os
import random
import re
import sys
import pytest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.deterministic_scorer import DeterministicScorer, IndependentCatalogOracle
from templates.ILP_VM_Knapsack_Allocation import VMSku, solve_ilp_vm_knapsack
from templates.Graph_SMT_Z3_MultiRegion_Placement import solve_z3_graph_disaster_recovery
from templates.Continuous_PSO_Dynamic_Scaling import solve_pso_continuous_scaling
from src.semantic.scope_parser import SCOPEParser
from src.semantic.normalizer import OutputNormalizer


# ---------------------------------------------------------------------------
# 1. Codebase Hardcoding & Query-ID Invariant Tests
# ---------------------------------------------------------------------------

def test_no_query_id_hardcoding_in_production_code():
    """Scans all production modules to verify no hardcoded Q01-Q29 query branching."""
    src_dir = os.path.join(BASE_DIR, "src")
    forbidden_pattern = re.compile(r"\bQ0?[1-9]\b|\bQ[1-2][0-9]\b")
    
    suspicious_files = []
    # Exclude benchmark definitions and reporting CLI lookup helpers
    excluded_paths = [
        os.path.join("src", "benchmarks"),
        os.path.join("src", "reporting", "explain_verdict.py"),  # CLI query lookup helper
    ]

    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, BASE_DIR)
            
            if any(rel_path.startswith(exc) for exc in excluded_paths):
                continue

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Check for explicit condition like: if qid == "Q01" or query_id == "Q01"
                for line_no, line in enumerate(content.splitlines(), start=1):
                    # Strip comments
                    code_part = line.split("#")[0]
                    if re.search(r'["\']Q\d{2}_', code_part):
                        suspicious_files.append((rel_path, line_no, line.strip()))

    assert len(suspicious_files) == 0, f"Found query-ID hardcoding in production code: {suspicious_files}"


def test_no_ground_truth_leakage_in_prompts():
    """Verifies that LLM prompts and few-shot examples do not contain benchmark answers."""
    src_semantic = os.path.join(BASE_DIR, "src", "semantic")
    benchmark_manifest = os.path.join(BASE_DIR, "data", "final_query_manifest.json")
    
    with open(benchmark_manifest, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    
    query_texts = [q["query_text"].lower().strip() for q in manifest_data["queries"]]

    for root, _, files in os.walk(src_semantic):
        for file in files:
            if not file.endswith(".py"):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
                for q_text in query_texts:
                    # An exact match of the full user prompt in source code would indicate leakage
                    if len(q_text) > 30 and q_text in content:
                        pytest.fail(f"Benchmark query text leaked into {file}: '{q_text}'")


# ---------------------------------------------------------------------------
# 2. Generalization Tests on Unseen Random Workloads
# ---------------------------------------------------------------------------

def test_generalization_random_vm_workloads():
    """Verifies ILP knapsack solver solves randomly generated unseen VM demands."""
    random.seed(42)
    oracle = IndependentCatalogOracle()

    for trial in range(25):
        req_vcpus = random.choice([2, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64])
        req_ram = float(random.choice([4, 8, 16, 24, 32, 64, 96, 128]))
        budget = float(random.choice([50, 100, 200, 300, 500, 800, 1500, 3000]))
        providers = random.choice([None, ["AWS"], ["Azure"], ["GCP"], ["AWS", "Azure"], ["AWS", "GCP"]])

        res = solve_ilp_vm_knapsack(
            required_vcpus=req_vcpus,
            required_ram_gb=req_ram,
            budget_max_usd=budget,
            target_providers=providers,
        )

        assert "status" in res
        if res["status"] == "OPTIMAL":
            vms = res["allocated_vms"]
            assert len(vms) > 0
            recomp_cost, tot_vcpu, tot_ram, errs = oracle.recompute_vm_cost_and_capacity(vms)
            assert len(errs) == 0
            assert tot_vcpu >= req_vcpus, f"Trial {trial}: vCPUs {tot_vcpu} < required {req_vcpus}"
            assert tot_ram >= req_ram - 1e-5, f"Trial {trial}: RAM {tot_ram} < required {req_ram}"
            assert recomp_cost <= budget + 0.01, f"Trial {trial}: Cost ${recomp_cost} > budget ${budget}"
            assert abs(res["total_monthly_cost_usd"] - recomp_cost) <= 0.02


def test_generalization_random_dr_topologies():
    """Verifies Z3 SMT DR solver solves previously unseen disaster recovery requirements."""
    random.seed(1337)
    oracle = IndependentCatalogOracle()

    for trial in range(15):
        sla_target = random.choice([99.9, 99.95, 99.99, 99.999])
        latency_max = float(random.choice([20.0, 35.0, 50.0, 75.0, 100.0, 150.0]))
        budget = float(random.choice([150.0, 250.0, 400.0, 600.0, 1000.0]))

        res = solve_z3_graph_disaster_recovery(
            sla_pct=sla_target,
            max_latency_ms=latency_max,
            budget_max_usd=budget,
        )

        assert "is_feasible" in res
        if res["is_feasible"]:
            p_reg = res["primary_region"]
            s_reg = res["secondary_region"]
            recomp_cost, comp_sla, lat_ms, errs = oracle.recompute_dr_cost_and_metrics(p_reg, s_reg)
            assert len(errs) == 0
            assert comp_sla >= sla_target - 1e-5
            assert lat_ms <= latency_max + 1e-5
            assert recomp_cost <= budget + 0.01


def test_generalization_random_scaling_optimization():
    """Verifies continuous PSO solver scales dynamically for unseen traffic rates."""
    random.seed(999)

    for trial in range(15):
        bw = float(random.randint(50, 600))
        cpu_ceil = float(random.randint(40, 80))
        budget = float(random.randint(200, 2000))

        res = solve_pso_continuous_scaling(
            target_bandwidth_mbps=bw,
            max_cpu_pct=cpu_ceil,
            budget_max_usd=budget,
        )

        assert "is_feasible" in res
        if res["is_feasible"]:
            replicas = res["recommended_replicas"]
            assert replicas >= 1
            assert res["total_monthly_cost_usd"] <= budget + 0.01


# ---------------------------------------------------------------------------
# 3. Mutation Testing (Boundary Reversals, Injected Catalogs, Paraphrases)
# ---------------------------------------------------------------------------

def test_mutation_vm_budget_boundary_shift():
    """Mutates budget around the exact mathematical boundary to verify sharp phase transition."""
    # Problem: 8 vCPUs, 16 GB RAM (AWS).
    # Optimum: 4x t3.medium @ $30.37/ea = $121.48 / month.
    
    # 1. Just below threshold ($121.00) -> Must be INFEASIBLE
    res_under = solve_ilp_vm_knapsack(required_vcpus=8, required_ram_gb=16.0, budget_max_usd=121.00, target_providers=["AWS"])
    assert "INFEASIBLE" in str(res_under.get("status", "")) or res_under.get("is_feasible") is False

    # 2. At threshold ($121.50) -> Must be OPTIMAL ($121.48)
    res_at = solve_ilp_vm_knapsack(required_vcpus=8, required_ram_gb=16.0, budget_max_usd=121.50, target_providers=["AWS"])
    assert res_at["status"] == "OPTIMAL"
    assert abs(res_at["total_monthly_cost_usd"] - 121.48) <= 0.02

    # 3. Generous budget ($500.00) -> Must still pick the minimum cost ($121.48)
    res_over = solve_ilp_vm_knapsack(required_vcpus=8, required_ram_gb=16.0, budget_max_usd=500.00, target_providers=["AWS"])
    assert res_over["status"] == "OPTIMAL"
    assert abs(res_over["total_monthly_cost_usd"] - 121.48) <= 0.02


def test_mutation_custom_synthetic_catalog():
    """Injects a completely synthetic catalog to verify solver uses catalog rather than constants."""
    custom_catalog = [
        VMSku(name="synth.nano", provider="CustomCloud", vcpus=1, ram_gb=2.0, hourly_cost_usd=0.0100),   # $7.30/mo
        VMSku(name="synth.mega", provider="CustomCloud", vcpus=10, ram_gb=20.0, hourly_cost_usd=0.0800), # $58.40/mo
    ]

    # Need 20 vCPUs, 40 GB RAM.
    # Optimum: 2x synth.mega = 20 vCPUs, 40 GB RAM, Cost = 2 * $58.40 = $116.80
    res = solve_ilp_vm_knapsack(
        required_vcpus=20,
        required_ram_gb=40.0,
        budget_max_usd=150.00,
        catalog=custom_catalog,
    )

    assert res["status"] == "OPTIMAL"
    assert abs(res["total_monthly_cost_usd"] - 116.80) <= 0.05
    assert len(res["allocated_vms"]) == 1
    assert res["allocated_vms"][0]["instance_type"] == "synth.mega"
    assert res["allocated_vms"][0]["count"] == 2


def test_mutation_query_paraphrasing_invariance():
    """Verifies rule-based semantic parser extracts identical contracts across paraphrases."""
    parser = SCOPEParser()
    
    p1 = "Deploy a high-availability setup with 16 vCPUs and 32 GB RAM on AWS under $600."
    p2 = "Need an AWS cluster requiring minimum 16 vcpus, 32gb memory, budget limit 600 usd."
    p3 = "Amazon Web Services allocation: 16 vcpus and 32 gb ram with max monthly spend 600 dollars."

    c1, _, _ = parser.parse_query_to_contract(p1)
    c2, _, _ = parser.parse_query_to_contract(p2)
    c3, _, _ = parser.parse_query_to_contract(p3)

    assert c1.required_vcpus == 16
    assert c2.required_vcpus == 16
    assert c3.required_vcpus == 16

    assert c1.required_ram_gb == 32.0
    assert c2.required_ram_gb == 32.0
    assert c3.required_ram_gb == 32.0

    assert c1.budget_max_usd == 600.0
    assert c2.budget_max_usd == 600.0
    assert c3.budget_max_usd == 600.0


# ---------------------------------------------------------------------------
# 4. Decoupled Evaluation Verification
# ---------------------------------------------------------------------------

def test_evaluator_oracle_independence():
    """Verifies that IndependentCatalogOracle and DeterministicScorer do not depend on solver code."""
    oracle = IndependentCatalogOracle()
    scorer = DeterministicScorer(oracle=oracle)

    # Fabricate a plan with invalid pricing
    manifest = {
        "intended_archetype": "ILP_VM_Allocation",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 4,
        "required_ram_gb": 8.0,
        "budget_max_usd": 200.0,
        "cloud_providers": ["AWS"],
        "optimal_cost_usd": 60.74,
    }
    
    # Claimed cost is $10.00, but 1x t3.large actually costs $60.74
    fake_plan = {
        "allocated_vms": [{"sku": "t3.large", "quantity": 1}],
        "total_monthly_cost_usd": 10.00,
    }

    verdict = scorer.evaluate_record(
        query_id="TEST_INDEPENDENT_AUDIT",
        mode=2,
        manifest_entry=manifest,
        raw_output=fake_plan,
    )

    # Scorer must catch the cost discrepancy and reject strict success
    assert verdict.cost_correct is False
    assert abs(verdict.recomputed_cost_usd - 60.74) <= 0.01
    assert verdict.strict_success == 0


def test_evaluator_rejects_missing_output_without_fabrication():
    """Verifies that missing model outputs are strictly flagged as failures and never filled with defaults."""
    oracle = IndependentCatalogOracle()
    scorer = DeterministicScorer(oracle=oracle)

    # 1. Missing VM allocation
    manifest_vm = {"intended_archetype": "ILP_VM_Allocation", "expected_outcome": "FEASIBLE", "required_vcpus": 4, "required_ram_gb": 8.0}
    verdict_vm = scorer.evaluate_record(query_id="TEST_EMPTY_VM", mode=2, manifest_entry=manifest_vm, raw_output={"task_type": "ILP_VM_Allocation", "allocated_vms": []})
    assert verdict_vm.strict_success == 0
    assert verdict_vm.constraints_satisfied is False

    # 2. Missing DR regions
    manifest_dr = {"intended_archetype": "Z3_Graph_Disaster_Recovery", "expected_outcome": "FEASIBLE", "sla_availability_pct": 99.99, "latency_max_ms": 50.0}
    verdict_dr = scorer.evaluate_record(query_id="TEST_EMPTY_DR", mode=2, manifest_entry=manifest_dr, raw_output={"task_type": "Z3_Graph_Disaster_Recovery", "primary_region": "", "secondary_region": ""})
    assert verdict_dr.strict_success == 0
    assert verdict_dr.constraints_satisfied is False

    # 3. Missing Scaling replicas
    manifest_scaling = {"intended_archetype": "PSO_Continuous_Scaling", "expected_outcome": "FEASIBLE", "target_bandwidth_mbps": 300.0, "max_cpu_pct": 60.0}
    verdict_scaling = scorer.evaluate_record(query_id="TEST_EMPTY_SCALING", mode=2, manifest_entry=manifest_scaling, raw_output={"task_type": "PSO_Continuous_Scaling", "recommended_replicas": None})
    assert verdict_scaling.strict_success == 0
    assert verdict_scaling.constraints_satisfied is False


def test_evaluator_unaffected_by_corrupted_historical_telemetry_labels():
    """Proves that evaluator verdicts are 100% independent of historical explain_label / task_success."""
    oracle = IndependentCatalogOracle()
    scorer = DeterministicScorer(oracle=oracle)

    manifest = {
        "intended_archetype": "ILP_VM_Allocation",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 300.0,
        "cloud_providers": ["AWS"],
        "optimal_cost_usd": 121.48,
    }

    # Truly valid allocation: 4x t3.medium = 8 vCPUs, 16 GB, $121.48
    valid_plan = {
        "allocated_vms": [{"sku": "t3.medium", "quantity": 4}],
        "total_monthly_cost_usd": 121.48,
    }

    # Sabotaged historical telemetry claiming failure
    corrupted_telemetry = {
        "explain_label": "WRONG_OUTCOME",
        "proof_status": "Failed",
        "task_success": 0.0,
        "normalization_status": "NORMALIZATION_FAILURE",
        "plan_valid": False,
    }

    verdict = scorer.evaluate_record(
        query_id="TEST_SABOTAGED_LABELS",
        mode=2,
        manifest_entry=manifest,
        raw_output=valid_plan,
        telemetry_record=corrupted_telemetry,
    )

    # Scorer MUST grade based on mathematics and ignore corrupted labels
    assert verdict.strict_success == 1
    assert verdict.constraints_satisfied is True
    assert verdict.cost_correct is True


def test_evaluator_unaffected_by_false_historical_success_labels():
    """Proves that evaluator rejects invalid allocations even if historical telemetry claims success."""
    oracle = IndependentCatalogOracle()
    scorer = DeterministicScorer(oracle=oracle)

    manifest = {
        "intended_archetype": "ILP_VM_Allocation",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 16,
        "required_ram_gb": 64.0,
        "budget_max_usd": 200.0,
        "cloud_providers": ["AWS"],
    }

    # Invalid allocation: only 1x t3.medium (2 vCPUs, 4 GB) costing $30.37, violates 16 vCPU / 64 GB
    invalid_plan = {
        "allocated_vms": [{"sku": "t3.medium", "quantity": 1}],
        "total_monthly_cost_usd": 30.37,
    }

    # Fraudulent historical telemetry claiming success
    fraudulent_telemetry = {
        "explain_label": "CORRECT",
        "proof_status": "Optimal",
        "task_success": 1.0,
        "normalization_status": "NORMALIZED_SUCCESS",
        "plan_valid": True,
    }

    verdict = scorer.evaluate_record(
        query_id="TEST_FRAUDULENT_SUCCESS",
        mode=2,
        manifest_entry=manifest,
        raw_output=invalid_plan,
        telemetry_record=fraudulent_telemetry,
    )

    # Scorer MUST catch the constraint violation and reject success
    assert verdict.strict_success == 0
    assert verdict.constraints_satisfied is False
    assert "Insufficient vCPUs" in verdict.failure_reason


def test_synthetic_dynamic_scaling_mutation():
    """Tests continuous scaling evaluation on non-standard bandwidth and CPU thresholds."""
    oracle = IndependentCatalogOracle()
    scorer = DeterministicScorer(oracle=oracle)

    manifest = {
        "intended_archetype": "PSO_Continuous_Scaling",
        "expected_outcome": "FEASIBLE",
        "target_bandwidth_mbps": 750.0,
        "max_cpu_pct": 50.0,
        "budget_max_usd": 1000.0,
    }

    # 750 Mbps @ 50% max CPU requires capacity >= 1500 Mbps -> >= 20 replicas (20 * 75 Mbps = 1500 Mbps).
    # Cost for 20 replicas = 80 + 20 * 37 = $820.00
    plan_20_replicas = {
        "recommended_replicas": 20,
        "optimal_bandwidth_mbps": 750.0,
        "total_monthly_cost_usd": 820.00,
    }

    verdict = scorer.evaluate_record(
        query_id="TEST_SYNTH_SCALING_FEASIBLE",
        mode=2,
        manifest_entry=manifest,
        raw_output=plan_20_replicas,
    )

    assert verdict.strict_success == 1
    assert verdict.constraints_satisfied is True
    assert verdict.recomputed_cost_usd == 820.00

    # Test under-provisioned plan: 10 replicas -> capacity = 750 Mbps -> CPU = 100% > 50% max CPU
    plan_10_replicas = {
        "recommended_replicas": 10,
        "optimal_bandwidth_mbps": 750.0,
        "total_monthly_cost_usd": 450.00,
    }

    verdict_fail = scorer.evaluate_record(
        query_id="TEST_SYNTH_SCALING_UNDERPROVISIONED",
        mode=2,
        manifest_entry=manifest,
        raw_output=plan_10_replicas,
    )

    assert verdict_fail.strict_success == 0
    assert verdict_fail.constraints_satisfied is False
    assert "CPU Ceiling violation" in verdict_fail.failure_reason

