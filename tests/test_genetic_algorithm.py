import pytest
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm

def test_ga_initialization_bounds():
    ga = GeneticAlgorithm(population_size=10, generations=2, random_seed=42)
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )
    res = ga.solve(req)
    assert res.solver_name == "Vectorized_GA"
    # Even if it finds or doesn't find feasible, it shouldn't crash and should return Result
    assert hasattr(res, "is_feasible")
    assert res.metrics.iterations == 2

def test_ga_reproducibility():
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=500.0,
        service_count=1,
        required_vcpus=8,
        required_ram_gb=16.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )
    ga1 = GeneticAlgorithm(population_size=20, generations=10, random_seed=123)
    res1 = ga1.solve(req)
    
    ga2 = GeneticAlgorithm(population_size=20, generations=10, random_seed=123)
    res2 = ga2.solve(req)
    
    if res1.best_candidate and res2.best_candidate:
        assert res1.best_candidate.objective_cost_usd == res2.best_candidate.objective_cost_usd

def test_ga_infeasible_request():
    # Impossible budget
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=0.01,
        service_count=1,
        required_vcpus=100,
        required_ram_gb=1000.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )
    ga = GeneticAlgorithm(population_size=10, generations=5)
    res = ga.solve(req)
    assert not res.is_feasible
