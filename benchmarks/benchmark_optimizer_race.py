import time
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
    
    print(f"--- Running SYM-1 Optimizer Race Benchmark ---")
    
    # Measure GA independently
    start = time.perf_counter()
    ga_res = ga.solve(req)
    ga_time = (time.perf_counter() - start) * 1000.0
    print(f"Vectorized GA: {ga_time:.2f} ms | Feasible: {ga_res.is_feasible} | Cost: ${ga_res.best_candidate.objective_cost_usd if ga_res.best_candidate else 'N/A'}")
    
    # Measure PSO independently
    start = time.perf_counter()
    pso_res = pso.solve(req)
    pso_time = (time.perf_counter() - start) * 1000.0
    print(f"Vectorized PSO: {pso_time:.2f} ms | Feasible: {pso_res.is_feasible} | Cost: ${pso_res.best_candidate.objective_cost_usd if pso_res.best_candidate else 'N/A'}")
    
    # Measure sequential race
    start = time.perf_counter()
    race_res = race.execute_race(req)
    race_time = (time.perf_counter() - start) * 1000.0
    print(f"Sequential Race Total: {race_time:.2f} ms | Winner: {race_res.solver_name} | Cost: ${race_res.best_candidate.objective_cost_usd if race_res.best_candidate else 'N/A'}")

if __name__ == "__main__":
    run_benchmark()
