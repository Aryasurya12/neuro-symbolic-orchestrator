import time
import json
import logging
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.optihive.solver_selection import OptiHiveSelector
from src.symbolic.data.repository import CloudDataRepository
from src.symbolic.data.seed import seed_database
import numpy as np

# Suppress debug logs
logging.basicConfig(level=logging.ERROR)

def evaluate_system():
    print("============================================================")
    print("SYSTEM EVALUATION: Phase 7")
    print("============================================================")

    # 1. Database Evaluation
    print("\n[Database Evaluation]")
    t0 = time.perf_counter()
    repo = CloudDataRepository()
    t1 = time.perf_counter()
    print(f"Repository Init Latency: {(t1 - t0) * 1000:.2f} ms")
    
    t0 = time.perf_counter()
    skus = repo.get_skus()
    t1 = time.perf_counter()
    print(f"SKU Lookup Latency (Total: {len(skus)}): {(t1 - t0) * 1000:.2f} ms")
    
    t0 = time.perf_counter()
    feasible = repo.find_feasible_skus(required_vcpus=4, required_ram_gb=16.0)
    t1 = time.perf_counter()
    print(f"Feasibility Query Latency (Found: {len(feasible)}): {(t1 - t0) * 1000:.2f} ms")
    
    # Define requests
    experiments = {
        "Exp A (Small Feasible)": SymbolicOptimizationRequest(
            problem_type="VM_PLACEMENT",
            cloud_providers=["AWS", "Azure"],
            budget_max_usd=200.0,
            service_count=1,
            required_vcpus=2,
            required_ram_gb=4.0,
            latency_max_ms=100.0,
            sla_availability_pct=99.0
        ),
        "Exp B (Large Resource)": SymbolicOptimizationRequest(
            problem_type="VM_PLACEMENT",
            cloud_providers=["AWS", "GCP"],
            budget_max_usd=500.0,
            service_count=1,
            required_vcpus=8,
            required_ram_gb=32.0,
            latency_max_ms=100.0,
            sla_availability_pct=99.0
        ),
        "Exp C (Tight Budget)": SymbolicOptimizationRequest(
            problem_type="VM_PLACEMENT",
            cloud_providers=["AWS"],
            budget_max_usd=25.0, # Very low
            service_count=1,
            required_vcpus=2,
            required_ram_gb=4.0,
            latency_max_ms=100.0,
            sla_availability_pct=99.0
        ),
        "Exp E (Infeasible)": SymbolicOptimizationRequest(
            problem_type="VM_PLACEMENT",
            cloud_providers=["AWS", "Azure", "GCP"],
            budget_max_usd=10.0, # Impossible
            service_count=1,
            required_vcpus=16,
            required_ram_gb=64.0,
            latency_max_ms=10.0,
            sla_availability_pct=99.99
        ),
    }
    
    # 2. Hard Constraint Validation
    print("\n[Hard-Constraint Validation]")
    req_hard = experiments["Exp C (Tight Budget)"]
    print(f"Request: Budget <= {req_hard.budget_max_usd}, vCPUs >= {req_hard.required_vcpus}, RAM >= {req_hard.required_ram_gb}")
    ga = GeneticAlgorithm(random_seed=42)
    res_ga = ga.solve(req_hard)
    print(f"GA Feasible: {res_ga.is_feasible}, Cost: {res_ga.best_candidate.objective_cost_usd if res_ga.best_candidate else 'N/A'}")
    if res_ga.is_feasible:
        print("Violation! GA should not find a feasible solution for this budget.")
    else:
        print("Success: GA correctly identified infeasibility. Soft preferences did not override hard constraints.")
        
        print("\n[OptiHive Pipeline Evaluation]")
    selector = OptiHiveSelector(seed=42)
    
    for name, req in experiments.items():
        print(f"\n--- {name} ---")
        # Run solvers
        ga_solver = GeneticAlgorithm(random_seed=42)
        pso_solver = ParticleSwarmOptimization(random_seed=42)
        z3_solver = GraphSteeredZ3Solver()
        
        t0 = time.perf_counter()
        ga_res = ga_solver.solve(req)
        ga_time = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        pso_res = pso_solver.solve(req)
        pso_time = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        z3_res = z3_solver.solve(req)
        z3_time = (time.perf_counter() - t0) * 1000
        
        print(f"GA  -> Feasible: {ga_res.is_feasible}, Time: {ga_time:.2f}ms, Cost: {ga_res.best_candidate.objective_cost_usd if ga_res.is_feasible else 'N/A'}")
        print(f"PSO -> Feasible: {pso_res.is_feasible}, Time: {pso_time:.2f}ms, Cost: {pso_res.best_candidate.objective_cost_usd if pso_res.is_feasible else 'N/A'}")
        print(f"Z3  -> Feasible: {z3_res.is_feasible}, Time: {z3_time:.2f}ms, Cost: {z3_res.best_candidate.objective_cost_usd if z3_res.is_feasible else 'N/A'}")
        
        # OptiHive Selection
        candidates = [ga_res, pso_res, z3_res]
        
        t0 = time.perf_counter()
        selected = selector.select(req, candidates)
        sel_time = (time.perf_counter() - t0) * 1000
        
        num_input = len(candidates)
        num_feasible = sum(1 for c in candidates if c.is_feasible)
        
        print(f"OptiHive -> Input: {num_input}, Feasible: {num_feasible}, Selection Time: {sel_time:.2f}ms")
        if selected and selected.best_candidate:
            print(f"OptiHive Selected: {selected.solver_name} with Cost {selected.best_candidate.objective_cost_usd}")
        else:
            print("OptiHive Selected: None (All infeasible)")

if __name__ == "__main__":
    evaluate_system()
