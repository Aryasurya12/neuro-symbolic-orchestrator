import time
import json
import logging
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.optihive.solver_selection import OptiHiveSelector

# Suppress debug logs
logging.basicConfig(level=logging.ERROR)

def run_demo():
    print("============================================================")
    print("END-TO-END SYSTEM DEMO (Phase 7)")
    print("============================================================\n")

    # 1. Incoming Request (Simulated from Part A Semantic Layer)
    req_dict = {
        "problem_type": "VM_PLACEMENT",
        "cloud_providers": ["AWS", "Azure"],
        "budget_max_usd": 300.0,
        "service_count": 1,
        "required_vcpus": 4,
        "required_ram_gb": 16.0,
        "latency_max_ms": 50.0,
        "sla_availability_pct": 99.9
    }
    
    print("[1] Received Semantic Request:")
    print(json.dumps(req_dict, indent=2))
    
    req = SymbolicOptimizationRequest(**req_dict)
    
    # 2. Parallel Execution of Symbolic Engines
    print("\n[2] Executing Symbolic Engines...")
    
    t0 = time.perf_counter()
    ga = GeneticAlgorithm(random_seed=42)
    res_ga = ga.solve(req)
    t_ga = time.perf_counter() - t0
    print(f"  -> GA finished in {t_ga*1000:.2f} ms")
    
    t0 = time.perf_counter()
    pso = ParticleSwarmOptimization(random_seed=42)
    res_pso = pso.solve(req)
    t_pso = time.perf_counter() - t0
    print(f"  -> PSO finished in {t_pso*1000:.2f} ms")
    
    t0 = time.perf_counter()
    z3s = GraphSteeredZ3Solver()
    res_z3 = z3s.solve(req)
    t_z3 = time.perf_counter() - t0
    print(f"  -> Z3 finished in {t_z3*1000:.2f} ms")
    
    # 3. OptiHive-Inspired Selection
    print("\n[3] OptiHive Latent Selection...")
    candidates = [res_ga, res_pso, res_z3]
    selector = OptiHiveSelector(seed=42)
    
    t0 = time.perf_counter()
    final_result = selector.select(req, candidates)
    t_opti = time.perf_counter() - t0
    print(f"  -> Selection finished in {t_opti*1000:.2f} ms")
    
    # 4. Final Output
    print("\n============================================================")
    print("FINAL OPTIMIZATION RESULT")
    print("============================================================")
    if final_result and final_result.is_feasible:
        print(f"Selected Solver : {final_result.solver_name}")
        print(f"Total Runtime   : {final_result.metrics.runtime_ms:.2f} ms")
        print(f"Best Cost (USD) : ${final_result.best_candidate.objective_cost_usd:.2f} / month")
        print("Decision Variables:")
        print(json.dumps(final_result.best_candidate.decision_variables, indent=2))
        if final_result.best_candidate.constraint_status:
            status = final_result.best_candidate.constraint_status
            print("Constraint Verification:")
            print(f"  - Budget OK: {status.budget_ok}")
            print(f"  - vCPU OK  : {status.vcpu_ok}")
            print(f"  - RAM OK   : {status.ram_ok}")
    else:
        print("No feasible candidate found across all solvers.")

if __name__ == "__main__":
    run_demo()
