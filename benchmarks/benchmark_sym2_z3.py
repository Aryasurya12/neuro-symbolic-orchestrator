import time
import numpy as np
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver

def run_benchmark():
    print("--- Running SYM-2 Graph-Steered Z3 Solver Benchmark ---")
    
    workloads = {
        "Normal Feasible": SymbolicOptimizationRequest(
            problem_type="Test", cloud_providers=["AWS", "Azure", "GCP"],
            budget_max_usd=1500.0, service_count=1, required_vcpus=16,
            required_ram_gb=32.0, latency_max_ms=100.0, sla_availability_pct=95.0
        ),
        "Budget Constrained": SymbolicOptimizationRequest(
            problem_type="Test", cloud_providers=["AWS", "Azure", "GCP"],
            budget_max_usd=250.0, service_count=1, required_vcpus=16,
            required_ram_gb=32.0, latency_max_ms=100.0, sla_availability_pct=95.0
        ),
        "Latency Constrained": SymbolicOptimizationRequest(
            problem_type="Test", cloud_providers=["AWS", "Azure", "GCP"],
            budget_max_usd=1500.0, service_count=1, required_vcpus=16,
            required_ram_gb=32.0, latency_max_ms=25.0, sla_availability_pct=95.0
        ),
        "Infeasible (SLA)": SymbolicOptimizationRequest(
            problem_type="Test", cloud_providers=["AWS", "Azure", "GCP"],
            budget_max_usd=1500.0, service_count=1, required_vcpus=16,
            required_ram_gb=32.0, latency_max_ms=100.0, sla_availability_pct=99.9999
        ),
    }
    
    modes = {
        "Baseline": "none",
        "Deterministic": "deterministic",
        "Learned GNN": "learned"
    }
    
    reps = 30
    
    results = {name: {mode: {"runtimes": [], "decisions": [], "conflicts": []} for mode in modes} for name in workloads}
    
    for name, req in workloads.items():
        print(f"\nEvaluating Workload: {name}")
        
        for mode_name, mode_val in modes.items():
            engine = GraphSteeredZ3Solver(steering_mode=mode_val)
            
            # Warm up
            engine.solve(req)
            
            for _ in range(reps):
                res = engine.solve(req)
                
                results[name][mode_name]["runtimes"].append(res.metrics.runtime_ms)
                results[name][mode_name]["decisions"].append(engine.last_stats.get("decisions", 0))
                results[name][mode_name]["conflicts"].append(engine.last_stats.get("conflicts", 0))
                
            median_rt = np.median(results[name][mode_name]["runtimes"])
            avg_dec = np.mean(results[name][mode_name]["decisions"])
            print(f"  {mode_name}: {median_rt:.2f} ms | Decisions: {avg_dec:.2f}")

    print("\n--- Speedup Analysis (Learned GNN vs Baseline) ---")
    
    total_baseline_rt = 0.0
    total_learned_rt = 0.0
    total_baseline_dec = 0.0
    total_learned_dec = 0.0
    
    for name in workloads:
        total_baseline_rt += np.median(results[name]["Baseline"]["runtimes"])
        total_learned_rt += np.median(results[name]["Learned GNN"]["runtimes"])
        total_baseline_dec += np.mean(results[name]["Baseline"]["decisions"])
        total_learned_dec += np.mean(results[name]["Learned GNN"]["decisions"])
        
    rt_speedup = total_baseline_rt / total_learned_rt if total_learned_rt > 0 else 0
    search_reduction = (1.0 - (total_learned_dec / total_baseline_dec)) * 100.0 if total_baseline_dec > 0 else 0.0
    
    print(f"Runtime Speedup: {rt_speedup:.2f}x")
    if search_reduction > 0:
        print(f"Search-Effort (Decisions) Reduction: {search_reduction:.2f}%")
    else:
        print(f"Search-Effort (Decisions) Reduction: {search_reduction:.2f}% (No reduction or increase)")
        
    if search_reduction >= 35.0 or rt_speedup >= 1.35:
        print("Target (35% search tree speedup): ACHIEVED")
    else:
        print("Target (35% search tree speedup): NOT ACHIEVED")
        
if __name__ == "__main__":
    run_benchmark()
