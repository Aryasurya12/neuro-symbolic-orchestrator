"""SYM-3: Adaptive Solver Routing Benchmark."""

import time
import json
from copy import deepcopy

from src.symbolic.models import SymbolicOptimizationRequest
from src.api.service import OptimizationService

# We can create a version of the service that DOES NOT use adaptive routing to compare against
class BaselineOptimizationService(OptimizationService):
    def run_pipeline_sync(self, request, progress_callback=None):
        from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
        from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
        from src.symbolic.optimizers.optimizer_race import OptimizerRace
        from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
        from src.symbolic.optihive.solver_selection import OptiHiveSelector
        
        start_time = time.perf_counter()
        
        ga_engine = GeneticAlgorithm(population_size=50, generations=20)
        pso_engine = ParticleSwarmOptimization(swarm_size=20, iterations=20)
        z3_engine = GraphSteeredZ3Solver(timeout_ms=2000)
        
        race = OptimizerRace(ga_engine=ga_engine, pso_engine=pso_engine)
        results = []
        
        race_result = race.execute_race(request, progress_callback=progress_callback)
        results.append(race_result)
        z3_result = z3_engine.solve(request, progress_callback=progress_callback)
        results.append(z3_result)
        
        selector = OptiHiveSelector()
        final_result = selector.select(request, results)
        
        total_time = (time.perf_counter() - start_time) * 1000.0
        # attach total benchmark runtime
        if final_result.best_candidate:
            final_result.best_candidate.metadata = final_result.best_candidate.metadata or {}
            final_result.best_candidate.metadata["benchmark_runtime"] = total_time
            
        return final_result


def run_benchmark():
    scenarios = [
        # 1. Normal discrete VM allocation
        SymbolicOptimizationRequest(
            problem_type="VM_ALLOCATION",
            cloud_providers=["aws", "gcp"],
            budget_max_usd=1000.0,
            service_count=5,
            required_vcpus=16,
            required_ram_gb=64.0,
            latency_max_ms=0,
            sla_availability_pct=0
        ),
        # 2. Large resource requirement
        SymbolicOptimizationRequest(
            problem_type="COMPUTE_INTENSIVE",
            cloud_providers=["aws", "azure", "gcp"],
            budget_max_usd=5000.0,
            service_count=15,
            required_vcpus=256,
            required_ram_gb=1024.0,
            latency_max_ms=0,
            sla_availability_pct=0
        ),
        # 3. Tight budget
        SymbolicOptimizationRequest(
            problem_type="COST_OPTIMIZATION",
            cloud_providers=["aws", "azure", "gcp"],
            budget_max_usd=100.0,
            service_count=3,
            required_vcpus=8,
            required_ram_gb=16.0,
            latency_max_ms=0,
            sla_availability_pct=0
        ),
        # 4. Infeasible request (Budget too low)
        SymbolicOptimizationRequest(
            problem_type="COST_OPTIMIZATION",
            cloud_providers=["aws"],
            budget_max_usd=5.0,
            service_count=10,
            required_vcpus=64,
            required_ram_gb=256.0,
            latency_max_ms=0,
            sla_availability_pct=0
        ),
        # 5. Multi-provider optimization
        SymbolicOptimizationRequest(
            problem_type="MULTI_CLOUD",
            cloud_providers=["aws", "azure", "gcp"],
            budget_max_usd=2000.0,
            service_count=8,
            required_vcpus=32,
            required_ram_gb=128.0,
            latency_max_ms=0,
            sla_availability_pct=0
        ),
        # 6. Latency constrained request
        SymbolicOptimizationRequest(
            problem_type="LOW_LATENCY",
            cloud_providers=["aws"],
            budget_max_usd=1500.0,
            service_count=4,
            required_vcpus=16,
            required_ram_gb=64.0,
            latency_max_ms=50.0,
            sla_availability_pct=0
        ),
        # 7. SLA constrained request
        SymbolicOptimizationRequest(
            problem_type="HIGH_AVAILABILITY",
            cloud_providers=["aws", "azure"],
            budget_max_usd=2500.0,
            service_count=6,
            required_vcpus=32,
            required_ram_gb=128.0,
            latency_max_ms=0,
            sla_availability_pct=99.99
        ),
        # 8. Topology/DR request
        SymbolicOptimizationRequest(
            problem_type="DISASTER_RECOVERY",
            cloud_providers=["aws", "azure", "gcp"],
            budget_max_usd=3000.0,
            service_count=10,
            required_vcpus=64,
            required_ram_gb=256.0,
            latency_max_ms=100.0,
            sla_availability_pct=99.9
        ),
        # 9. Small search space
        SymbolicOptimizationRequest(
            problem_type="SMALL_PROBLEM",
            cloud_providers=["aws"],
            budget_max_usd=500.0,
            service_count=1,
            required_vcpus=2,
            required_ram_gb=4.0,
            latency_max_ms=0,
            sla_availability_pct=0
        ),
        # 10. Larger search space
        SymbolicOptimizationRequest(
            problem_type="LARGE_PROBLEM",
            cloud_providers=["aws", "azure", "gcp", "oci"],
            budget_max_usd=10000.0,
            service_count=30,
            required_vcpus=1024,
            required_ram_gb=4096.0,
            latency_max_ms=200.0,
            sla_availability_pct=99.5
        ),
    ]

    baseline_service = BaselineOptimizationService()
    adaptive_service = OptimizationService()
    
    results_out = []
    
    print(f"{'Scenario':<25} | {'Feasible':<10} | {'Baseline (ms)':<15} | {'Adaptive (ms)':<15} | {'Solvers Used'}")
    print("-" * 100)
    
    for idx, req in enumerate(scenarios):
        # Run Baseline
        t0 = time.perf_counter()
        base_res = baseline_service.run_pipeline_sync(req)
        t_base = (time.perf_counter() - t0) * 1000.0
        
        # Run Adaptive
        t0 = time.perf_counter()
        adapt_res = adaptive_service.run_pipeline_sync(req)
        t_adapt = (time.perf_counter() - t0) * 1000.0
        
        routing_meta = {}
        solvers_used = "ALL (Baseline Fallback)"
        if adapt_res.best_candidate and "routing" in adapt_res.best_candidate.metadata:
            routing_meta = adapt_res.best_candidate.metadata["routing"]
            solvers_used = ",".join(routing_meta.get("selected_solvers", []))
            
        print(f"{req.problem_type:<25} | {str(adapt_res.is_feasible):<10} | {t_base:<15.2f} | {t_adapt:<15.2f} | {solvers_used}")
        
        results_out.append({
            "scenario": req.problem_type,
            "feasible": adapt_res.is_feasible,
            "baseline_ms": t_base,
            "adaptive_ms": t_adapt,
            "routing": routing_meta
        })

    with open("benchmark_sym3_routing.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print("Results saved to benchmark_sym3_routing.json")

if __name__ == "__main__":
    run_benchmark()
