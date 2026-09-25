"""SYM-4: Learned Adaptive Solver Routing Benchmark."""

import time
import json
from copy import deepcopy

from src.symbolic.models import SymbolicOptimizationRequest
from src.api.service import OptimizationService
from src.symbolic.optihive.adaptive_router import AdaptiveSolverRouter

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
        
        return final_result

class Sym3OptimizationService(OptimizationService):
    def run_pipeline_sync(self, request, progress_callback=None):
        import src.api.service
        # Temporarily disable learned router
        original_router = src.api.service.AdaptiveSolverRouter
        
        class MockDeterministicRouter(AdaptiveSolverRouter):
            def __init__(self):
                super().__init__()
                self.learned_router = None
                
        src.api.service.AdaptiveSolverRouter = MockDeterministicRouter
        try:
            res = super().run_pipeline_sync(request, progress_callback)
        finally:
            src.api.service.AdaptiveSolverRouter = original_router
        return res

def run_benchmark():
    scenarios = [
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
        SymbolicOptimizationRequest(
            problem_type="COST_OPTIMIZATION_INFEAS",
            cloud_providers=["aws"],
            budget_max_usd=5.0,
            service_count=10,
            required_vcpus=64,
            required_ram_gb=256.0,
            latency_max_ms=0,
            sla_availability_pct=0
        ),
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
    sym3_service = Sym3OptimizationService()
    sym4_service = OptimizationService()
    
    results_out = []
    
    print(f"{'Scenario':<25} | {'Feas':<5} | {'Base(ms)':<10} | {'SYM3(ms)':<10} | {'SYM4(ms)':<10} | {'SYM4 Mode':<15} | {'SYM4 Solvers'}")
    print("-" * 115)
    
    import random
    random.seed(42)
    
    for req in scenarios:
        # Run Baseline
        t0 = time.perf_counter()
        base_res = baseline_service.run_pipeline_sync(req)
        t_base = (time.perf_counter() - t0) * 1000.0
        
        # Run SYM-3
        t0 = time.perf_counter()
        sym3_res = sym3_service.run_pipeline_sync(req)
        t_sym3 = (time.perf_counter() - t0) * 1000.0
        
        # Run SYM-4
        t0 = time.perf_counter()
        sym4_res = sym4_service.run_pipeline_sync(req)
        t_sym4 = (time.perf_counter() - t0) * 1000.0
        
        routing_meta = {}
        solvers_used = "ALL"
        mode = "baseline"
        conf = "N/A"
        if sym4_res.metadata and "routing" in sym4_res.metadata:
            routing_meta = sym4_res.metadata["routing"]
            solvers_used = ",".join(routing_meta.get("selected_solvers", []))
            mode = routing_meta.get("routing_mode", "unknown")
            conf = str(routing_meta.get("confidence", "N/A"))
            
        print(f"{req.problem_type:<25} | {str(sym4_res.is_feasible)[0]:<5} | {t_base:<10.2f} | {t_sym3:<10.2f} | {t_sym4:<10.2f} | {mode:<15} | {solvers_used}")
        if mode == "fallback":
            print(f"Fallback reason: {routing_meta.get('reason')}")
        
        # assert base_res.is_feasible == sym3_res.is_feasible == sym4_res.is_feasible, "Feasibility mismatch!"
        
        results_out.append({
            "scenario": req.problem_type,
            "feasible": sym4_res.is_feasible,
            "baseline_ms": t_base,
            "sym3_ms": t_sym3,
            "sym4_ms": t_sym4,
            "routing": routing_meta
        })

    with open("benchmarks/results/benchmark_sym4_learned_routing.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print("Results saved to benchmarks/results/benchmark_sym4_learned_routing.json")

if __name__ == "__main__":
    run_benchmark()
