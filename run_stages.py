"""Neurasym: Neuro-Symbolic 6-Stage Pipeline Terminal Runner.

Visualizes each stage of the end-to-end neuro-symbolic pipeline in real time:
  Stage 1: State Encoding & Entity Extraction
  Stage 2: CARM Template Retrieval & Cosine Similarity Match
  Stage 3: Pydantic Contract Validation (JSON Handshake)
  Stage 4: Parallel Solver Race (GA vs PSO vs Graph-Steered Z3)
  Stage 5: OptiHive Latent-Class EM Solution Selection
  Stage 6: Natural Language Explainer & Proof Certificate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.semantic.scope_parser import SCOPEParser
from src.semantic.explainer import FinOpsExplainer
from src.symbolic.adapters import from_contract, to_explainer_dict
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.solvers.graph_model import InfrastructureGraph
from src.symbolic.solvers.graph_steering import GraphSteeringLayer
from src.symbolic.optihive.solver_selection import OptiHiveSelector


def print_header(title: str, stage_num: int):
    width = 80
    print("\n" + "=" * width)
    print(f"  ⚡ STAGE {stage_num}: {title.upper()}")
    print("=" * width)


def run_pipeline(user_query: str):
    print("\n" + "#" * 80)
    print("  🚀 NEURASYM NEURO-SYMBOLIC PIPELINE — TERMINAL EXECUTION MONITOR")
    print("#" * 80)
    print(f"  Input Query : \"{user_query}\"")
    print("#" * 80)

    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # STAGE 1: SCOPE Parser
    # -------------------------------------------------------------------------
    print_header("State Encoding & Parameter Extraction", 1)
    t0 = time.perf_counter()
    parser = SCOPEParser()
    contract, matched_template, score = parser.parse_query_to_contract(user_query)
    t_stage1 = (time.perf_counter() - t0) * 1000

    print(f"  • Extraction Status  : [SUCCESS] ({t_stage1:.2f} ms)")
    print(f"  • Problem Type       : {contract.problem_type}")
    print(f"  • Required vCPUs     : {contract.required_vcpus}")
    print(f"  • Required RAM (GB)  : {contract.required_ram_gb:.1f} GB")
    print(f"  • Budget Cap (USD)   : ${contract.budget_max_usd:.2f}")
    print(f"  • Target Providers   : {contract.cloud_providers}")
    print(f"  • SLA Availability   : {contract.sla_availability_pct:.2f}%")
    if contract.metadata:
        intent = contract.metadata.get("qualitative_intent")
        if intent:
            print(f"  • Qualitative Intent : {intent}")

    # -------------------------------------------------------------------------
    # STAGE 2: CARM Historical Match
    # -------------------------------------------------------------------------
    print_header("CARM Historical Match & Template Retrieval", 2)
    template_id = (
        matched_template.get("template_id", "N/A")
        if isinstance(matched_template, dict)
        else getattr(matched_template, "template_id", "N/A")
    )
    print(f"  • Matched Template ID: {template_id}")
    print(f"  • Cosine Similarity  : {score:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 3: Pydantic Contract JSON Handshake
    # -------------------------------------------------------------------------
    print_header("Pydantic JSON Contract Validation", 3)
    t0 = time.perf_counter()
    contract_json = contract.model_dump_json(indent=2)
    t_stage3 = (time.perf_counter() - t0) * 1000
    print(f"  • Contract Validation: VALID ({t_stage3:.2f} ms)")
    print("  • Serialized Safety Payload :")
    for line in contract_json.splitlines():
        print(f"      {line}")

    # -------------------------------------------------------------------------
    # STAGE 4: Parallel Solver Race (GA vs PSO vs Graph-Steered Z3)
    # -------------------------------------------------------------------------
    print_header("Parallel Symbolic Optimization Race", 4)
    sym_req = from_contract(contract)

    # Initialize Solvers
    graph = InfrastructureGraph()
    steering_layer = GraphSteeringLayer(graph)
    ga_solver = GeneticAlgorithm(random_seed=42)
    pso_solver = ParticleSwarmOptimization(random_seed=42)
    z3_solver = GraphSteeredZ3Solver(steering_layer=steering_layer, graph=graph)

    candidates = []

    # 1. GA Solver
    ga_res = ga_solver.solve(sym_req)
    candidates.append(ga_res)
    ga_icon = "✅" if ga_res.is_feasible else "❌"
    ga_cost = f"${ga_res.best_candidate.objective_cost_usd:.2f}" if ga_res.best_candidate else "N/A"
    print(f"\n  [Solver 1] Genetic Algorithm (GA) {ga_icon}")
    print(f"    - Feasible: {ga_res.is_feasible} | Monthly Cost: {ga_cost} | Runtime: {ga_res.metrics.runtime_ms:.2f}ms")
    if ga_res.best_candidate and "selected_vms" in ga_res.best_candidate.decision_variables:
        vms = ga_res.best_candidate.decision_variables["selected_vms"]
        for vm in vms:
            if isinstance(vm, dict):
                print(f"    - Allocated VM: {vm.get('name', vm.get('vm_id', 'vm'))} [{vm.get('provider', '')}] - {vm.get('vcpus')} vCPUs, {vm.get('ram_gb')}GB RAM (${vm.get('cost_per_hour', 0)*730:.2f}/mo)")

    # 2. PSO Solver
    pso_res = pso_solver.solve(sym_req)
    candidates.append(pso_res)
    pso_icon = "✅" if pso_res.is_feasible else "❌"
    pso_cost = f"${pso_res.best_candidate.objective_cost_usd:.2f}" if pso_res.best_candidate else "N/A"
    print(f"\n  [Solver 2] Particle Swarm Optimization (PSO) {pso_icon}")
    print(f"    - Feasible: {pso_res.is_feasible} | Monthly Cost: {pso_cost} | Runtime: {pso_res.metrics.runtime_ms:.2f}ms")
    if pso_res.best_candidate and "continuous_allocation" in pso_res.best_candidate.decision_variables:
        alloc = pso_res.best_candidate.decision_variables["continuous_allocation"]
        print(f"    - Allocation  : {alloc}")

    # 3. Z3 SMT Solver
    z3_res = z3_solver.solve(sym_req)
    candidates.append(z3_res)
    z3_icon = "✅" if z3_res.is_feasible else "❌"
    z3_cost = f"${z3_res.best_candidate.objective_cost_usd:.2f}" if z3_res.best_candidate else "N/A"
    print(f"\n  [Solver 3] Graph-Steered Z3 MaxSMT Solver {z3_icon}")
    print(f"    - Feasible: {z3_res.is_feasible} | Monthly Cost: {z3_cost} | Runtime: {z3_res.metrics.runtime_ms:.2f}ms")

    # -------------------------------------------------------------------------
    # STAGE 5: OptiHive Latent-Class EM Selection
    # -------------------------------------------------------------------------
    print_header("OptiHive Latent-Class EM Solver Selection", 5)
    t0 = time.perf_counter()
    selector = OptiHiveSelector(seed=42)
    final_result = selector.select(sym_req, candidates)
    t_stage5 = (time.perf_counter() - t0) * 1000

    if final_result and final_result.is_feasible and final_result.best_candidate:
        print(f"  • Winning Solver     : 🏆 {final_result.solver_name}")
        print(f"  • Mathematical Status: FEASIBLE (Zero Constraint Violations)")
        print(f"  • Optimal Spend      : ${final_result.best_candidate.objective_cost_usd:.2f} / mo (Budget: ${contract.budget_max_usd:.2f})")
        print(f"  • EM Scoring Latency : {t_stage5:.2f} ms")
        result_dict = to_explainer_dict(final_result, contract)
    else:
        print("  • Solver Selection   : INFEASIBLE (No valid candidates found)")
        primary_candidate = candidates[0] if candidates else None
        if primary_candidate:
            result_dict = to_explainer_dict(primary_candidate, contract)
        else:
            result_dict = {"status": "Infeasible", "solver": "OptiHive", "error_message": "No feasible candidate"}

    # -------------------------------------------------------------------------
    # STAGE 6: Natural Language Explainer & Proof Certificate
    # -------------------------------------------------------------------------
    print_header("Natural Language Explainer & Proof Certificate", 6)
    t0 = time.perf_counter()
    report = FinOpsExplainer.generate_report(contract, result_dict)
    t_stage6 = (time.perf_counter() - t0) * 1000

    print(f"  • Report Generation  : SUCCESS ({t_stage6:.2f} ms)")
    print("\n--- 📝 FINAL ALLOCATION REPORT ---")
    print(report)

    total_elapsed = (time.perf_counter() - total_start) * 1000
    print("\n" + "=" * 80)
    print(f"  🏁 END-TO-END 6-STAGE PIPELINE COMPLETE ({total_elapsed:.2f} ms)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Neurasym Neuro-Symbolic 6-Stage Pipeline Terminal Runner"
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default="Deploy a high compute workload with 8 vCPUs and 16GB RAM in AWS for under $300 a month with 99.9% SLA.",
        help="Natural language cloud infrastructure request.",
    )
    args = parser.parse_args()
    run_pipeline(args.query)


if __name__ == "__main__":
    main()
