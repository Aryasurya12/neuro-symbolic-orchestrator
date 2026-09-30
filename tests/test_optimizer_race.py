import pytest
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.optimizers.optimizer_race import OptimizerRace

def test_optimizer_race_execution():
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=500.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )
    
    ga = GeneticAlgorithm(population_size=10, generations=5, random_seed=42)
    pso = ParticleSwarmOptimization(swarm_size=10, iterations=5, random_seed=42)
    race = OptimizerRace(ga, pso)
    
    res = race.execute_race(req)
    
    # Depending on which one won, it should be fully populated
    assert res.solver_name in ["Vectorized_GA", "Vectorized_PSO"]
    assert res.metrics.runtime_ms > 0

def test_optimizer_race_infeasible_vs_feasible():
    # We can fake the results to test deterministic selection rule directly
    ga = GeneticAlgorithm()
    pso = ParticleSwarmOptimization()
    race = OptimizerRace(ga, pso)
    
    # Mock some results
    from src.symbolic.models import OptimizationResult, OptimizationCandidate, OptimizationMetrics, ConstraintStatus
    
    status_f = ConstraintStatus(is_feasible=True, budget_ok=True, vcpu_ok=True, ram_ok=True, latency_ok=True, sla_ok=True, details="")
    status_inf = ConstraintStatus(is_feasible=False, budget_ok=False, vcpu_ok=False, ram_ok=False, latency_ok=True, sla_ok=True, details="")
    
    cand_f = OptimizationCandidate(decision_variables={}, objective_cost_usd=100.0, is_feasible=True, constraint_status=status_f)
    cand_inf = OptimizationCandidate(decision_variables={}, objective_cost_usd=50.0, is_feasible=False, constraint_status=status_inf)
    
    res_f = OptimizationResult(best_candidate=cand_f, is_feasible=True, metrics=OptimizationMetrics(runtime_ms=10.0), solver_name="GA")
    res_inf = OptimizationResult(best_candidate=cand_inf, is_feasible=False, metrics=OptimizationMetrics(runtime_ms=5.0), solver_name="PSO")
    
    # Feasible should beat infeasible despite higher cost and higher runtime
    winner = race._select_best(res_f, res_inf)
    assert winner.solver_name == "GA"
    
    winner2 = race._select_best(res_inf, res_f)
    assert winner2.solver_name == "GA"

def test_optimizer_race_lower_cost_wins():
    ga = GeneticAlgorithm()
    pso = ParticleSwarmOptimization()
    race = OptimizerRace(ga, pso)
    
    from src.symbolic.models import OptimizationResult, OptimizationCandidate, OptimizationMetrics, ConstraintStatus
    
    status_f = ConstraintStatus(is_feasible=True, budget_ok=True, vcpu_ok=True, ram_ok=True, latency_ok=True, sla_ok=True, details="")
    cand_1 = OptimizationCandidate(decision_variables={}, objective_cost_usd=100.0, is_feasible=True, constraint_status=status_f)
    cand_2 = OptimizationCandidate(decision_variables={}, objective_cost_usd=80.0, is_feasible=True, constraint_status=status_f)
    
    res_1 = OptimizationResult(best_candidate=cand_1, is_feasible=True, metrics=OptimizationMetrics(runtime_ms=10.0), solver_name="GA")
    res_2 = OptimizationResult(best_candidate=cand_2, is_feasible=True, metrics=OptimizationMetrics(runtime_ms=20.0), solver_name="PSO")
    
    winner = race._select_best(res_1, res_2)
    assert winner.solver_name == "PSO"
