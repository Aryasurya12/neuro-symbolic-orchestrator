import pytest
import numpy as np
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization

def test_pso_initialization_bounds():
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
    pso = ParticleSwarmOptimization(swarm_size=10, iterations=5, random_seed=42)
    res = pso.solve(req)
    assert res.solver_name == "Vectorized_PSO"
    assert hasattr(res, "is_feasible")
    assert res.metrics.iterations == 5
    
    # If feasible, check bounds
    if res.is_feasible and res.best_candidate:
        bw = res.best_candidate.decision_variables.get("bandwidth_mbps", 0.0)
        reps = res.best_candidate.decision_variables.get("replicas", 0.0)
        assert 100.0 <= bw <= 1000.0
        assert 1.0 <= reps <= 16.0

def test_pso_reproducibility():
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
    pso1 = ParticleSwarmOptimization(swarm_size=20, iterations=10, random_seed=123)
    res1 = pso1.solve(req)
    
    pso2 = ParticleSwarmOptimization(swarm_size=20, iterations=10, random_seed=123)
    res2 = pso2.solve(req)
    
    if res1.best_candidate and res2.best_candidate:
        assert res1.best_candidate.objective_cost_usd == res2.best_candidate.objective_cost_usd

def test_pso_infeasible_request():
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=0.01,  # Impossible budget
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )
    pso = ParticleSwarmOptimization(swarm_size=10, iterations=5)
    res = pso.solve(req)
    assert not res.is_feasible
