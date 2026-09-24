import time
import numpy as np
import concurrent.futures
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.optimizers.optimizer_race import OptimizerRace

def run_benchmark():
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS", "Azure", "GCP"],
        budget_max_usd=1500.0,
        service_count=1,
        required_vcpus=16,
        required_ram_gb=32.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )
    
    ga = GeneticAlgorithm(population_size=100, generations=50, random_seed=42)
    pso = ParticleSwarmOptimization(swarm_size=50, iterations=50, random_seed=42)
    race = OptimizerRace(ga, pso)
    
    reps = 100
    ga_times = []
    pso_times = []
    parallel_times = []
    sequential_times = []
    
    print(f"--- Running SYM-1 Parallel Optimizer Race Benchmark ---")
    print("Warming up JIT/Caching...")
    ga.solve(req)
    pso.solve(req)
    race.execute_race(req)
    
    print(f"Executing {reps} repetitions...")
    
    for i in range(reps):
        # GA Alone
        start = time.perf_counter()
        ga_res = ga.solve(req)
        ga_times.append((time.perf_counter() - start) * 1000.0)
        
        # PSO Alone
        start = time.perf_counter()
        pso_res = pso.solve(req)
        pso_times.append((time.perf_counter() - start) * 1000.0)
        
        # Parallel Race
        start = time.perf_counter()
        race_res = race.execute_race(req)
        parallel_times.append((time.perf_counter() - start) * 1000.0)
        
        # Sequential Proxy
        # GA and PSO are independent; a strict sequential implementation
        # would just be ga_time + pso_time (excluding minor wrapper overhead).
        # We also execute them sequentially here for an exact measurement including wrappers.
        start = time.perf_counter()
        ga.solve(req)
        pso.solve(req)
        sequential_times.append((time.perf_counter() - start) * 1000.0)

    print("\n[Vectorized GA Alone]")
    print(f"  Median: {np.median(ga_times):.2f} ms")
    print(f"  Min:    {np.min(ga_times):.2f} ms")
    print(f"  Max:    {np.max(ga_times):.2f} ms")
    
    print("\n[Vectorized PSO Alone]")
    print(f"  Median: {np.median(pso_times):.2f} ms")
    print(f"  Min:    {np.min(pso_times):.2f} ms")
    print(f"  Max:    {np.max(pso_times):.2f} ms")
    
    print("\n[Sequential GA + PSO (Baseline)]")
    print(f"  Median: {np.median(sequential_times):.2f} ms")
    print(f"  Min:    {np.min(sequential_times):.2f} ms")
    print(f"  Max:    {np.max(sequential_times):.2f} ms")
    
    print("\n[Parallel GA + PSO (ThreadPool)]")
    print(f"  Median: {np.median(parallel_times):.2f} ms")
    print(f"  Min:    {np.min(parallel_times):.2f} ms")
    print(f"  Max:    {np.max(parallel_times):.2f} ms")
    print(f"  P95:    {np.percentile(parallel_times, 95):.2f} ms")
    
    speedup = np.median(sequential_times) / np.median(parallel_times)
    print(f"\n[Parallel Speedup]: {speedup:.2f}x")

    if np.median(parallel_times) < 1.0:
        print("\nTarget (Sub-millisecond): ACHIEVED")
    else:
        print("\nTarget (Sub-millisecond): NOT ACHIEVED")
        
if __name__ == "__main__":
    run_benchmark()
