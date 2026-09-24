import os
import json
import csv
import time
import statistics
from typing import Dict, Any, List
from src.orchestrator.service import NeuroSymbolicOrchestrator
from src.symbolic.models import OptimizationResult

# Configuration parameters
SEED = 42
REPETITIONS = 5

SCENARIOS = {
    "Normal Feasible": "Deploy a microservice needing 4 vCPUs and 16GB RAM on AWS or Azure for under $300 a month.",
    "Large Resource": "Deploy 128 vCPUs and 512GB RAM for under $3000.",
    "Tight Budget": "Deploy 8 vCPUs and 32GB RAM under $40.",
    "Intentionally Infeasible": "Deploy 64 vCPUs and 256GB RAM on Azure for under $15 a month.",
    "Multi-Provider": "Deploy 2 vCPUs on AWS, Azure, or GCP under $100.",
    "Latency/SLA Constrained (DR)": "Multi-region disaster recovery deployment with 99.99% SLA and under 20ms latency."
}

def measure_semantic_only(orchestrator: NeuroSymbolicOrchestrator, query: str) -> Dict[str, Any]:
    times = []
    contract = None
    for _ in range(REPETITIONS):
        t0 = time.perf_counter()
        contract, tmpl, score = orchestrator.parser.parse_query_to_contract(query)
        times.append((time.perf_counter() - t0) * 1000.0)
    
    return {
        "configuration": "Semantic-Only",
        "semantic_latency_ms": statistics.median(times),
        "symbolic_latency_ms": "N/A",
        "end_to_end_latency_ms": "N/A",
        "contract_generated": contract is not None,
        "is_feasible": "N/A",
        "cost_usd": "N/A",
        "solver": "N/A",
        "constraint_status": "N/A"
    }

def measure_symbolic_only(orchestrator: NeuroSymbolicOrchestrator, query: str, use_optihive: bool = True) -> Dict[str, Any]:
    contract, _, _ = orchestrator.parser.parse_query_to_contract(query)
    
    times = []
    res_dict = None
    for _ in range(REPETITIONS):
        t0 = time.perf_counter()
        if use_optihive:
            res_dict = orchestrator.optimize_contract(contract)
        else:
            # Bypass OptiHive
            from src.symbolic.adapters import from_contract
            from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
            sym_req = from_contract(contract)
            ga = GeneticAlgorithm(random_seed=SEED)
            res = ga.solve(sym_req)
            from src.symbolic.adapters import to_explainer_dict
            res_dict = to_explainer_dict(res, contract)
        times.append((time.perf_counter() - t0) * 1000.0)
        
    status = res_dict.get("constraint_status", {})
    return {
        "configuration": "Symbolic-Only" if use_optihive else "Symbolic-Only (No OptiHive)",
        "semantic_latency_ms": "N/A",
        "symbolic_latency_ms": statistics.median(times),
        "end_to_end_latency_ms": "N/A",
        "contract_generated": True,
        "is_feasible": res_dict.get("status") == "Feasible",
        "cost_usd": res_dict.get("total_monthly_cost_usd", "N/A"),
        "solver": res_dict.get("solver_name", "N/A"),
        "constraint_status": status
    }

def measure_neuro_symbolic(orchestrator: NeuroSymbolicOrchestrator, query: str) -> Dict[str, Any]:
    times = []
    report = None
    for _ in range(REPETITIONS):
        t0 = time.perf_counter()
        report = orchestrator.process_query(query)
        times.append((time.perf_counter() - t0) * 1000.0)
        
    contract, _, _ = orchestrator.parser.parse_query_to_contract(query)
    res_dict = orchestrator.optimize_contract(contract)
    
    status = res_dict.get("constraint_status", {})
    return {
        "configuration": "Neuro-Symbolic",
        "semantic_latency_ms": "N/A",
        "symbolic_latency_ms": "N/A",
        "end_to_end_latency_ms": statistics.median(times),
        "contract_generated": True,
        "is_feasible": res_dict.get("status") == "Feasible",
        "cost_usd": res_dict.get("total_monthly_cost_usd", "N/A"),
        "solver": res_dict.get("solver_name", "N/A"),
        "constraint_status": status
    }

def main():
    print("=" * 60)
    print("PHASE 9 - BASELINE & ABLATION EVALUATION")
    print("=" * 60)
    
    orchestrator = NeuroSymbolicOrchestrator()
    results = []
    
    for name, query in SCENARIOS.items():
        print(f"\nScenario: {name}")
        
        # 1. Semantic Only
        sem_res = measure_semantic_only(orchestrator, query)
        sem_res["scenario"] = name
        results.append(sem_res)
        print("Semantic-Only")
        print(f"  Contract generated: {'YES' if sem_res['contract_generated'] else 'NO'}")
        print(f"  Latency: {sem_res['semantic_latency_ms']:.2f} ms")
        
        # 2. Symbolic Only
        sym_res = measure_symbolic_only(orchestrator, query)
        sym_res["scenario"] = name
        results.append(sym_res)
        print("\nSymbolic-Only")
        print(f"  Feasible: {'YES' if sym_res['is_feasible'] else 'NO'}")
        cost_str = f"${sym_res['cost_usd']:.2f}" if sym_res['cost_usd'] != 'N/A' else 'N/A'
        print(f"  Cost: {cost_str}")
        print(f"  Latency: {sym_res['symbolic_latency_ms']:.2f} ms")
        
        # 3. Neuro-Symbolic
        ns_res = measure_neuro_symbolic(orchestrator, query)
        ns_res["scenario"] = name
        results.append(ns_res)
        print("\nNeuro-Symbolic")
        print(f"  Feasible: {'YES' if ns_res['is_feasible'] else 'NO'}")
        cost_str2 = f"${ns_res['cost_usd']:.2f}" if ns_res['cost_usd'] != 'N/A' else 'N/A'
        print(f"  Cost: {cost_str2}")
        print(f"  End-to-end latency: {ns_res['end_to_end_latency_ms']:.2f} ms")
        
        # 4. Symbolic-Only (No OptiHive Ablation)
        ab_res = measure_symbolic_only(orchestrator, query, use_optihive=False)
        ab_res["scenario"] = name
        results.append(ab_res)
        
    # Save Results
    os.makedirs("benchmarks/results", exist_ok=True)
    
    json_path = "benchmarks/results/phase9_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    csv_path = "benchmarks/results/phase9_baseline_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nSaved results to {json_path} and {csv_path}")

if __name__ == "__main__":
    main()
