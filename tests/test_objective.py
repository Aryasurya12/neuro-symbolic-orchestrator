import pytest
import numpy as np
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optimizers.objective import ObjectiveEvaluator
from src.symbolic.optimizers.domain_catalog import CATALOG_COSTS, CATALOG_VCPUS, CATALOG_RAM

def test_ga_objective_feasibility():
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=100.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )
    # Mock population of size 2, 10 SKUs
    # Pop 1: all 0s (cost 0, vcpu 0, ram 0) -> infeasible due to vcpu/ram
    # Pop 2: 10 units of SKU 0 (let's assume it breaks budget) -> infeasible due to budget
    pop = np.zeros((2, 10))
    pop[1, 0] = 1000 # Massive amount to break budget

    costs, is_feasible, constraints = ObjectiveEvaluator.evaluate_ga_population(pop, req)
    
    assert costs.shape == (2,)
    assert is_feasible.shape == (2,)
    
    # Pop 0 checks
    assert not constraints["vcpu_ok"][0]
    assert constraints["budget_ok"][0]
    assert not is_feasible[0]
    
    # Pop 1 checks
    assert not constraints["budget_ok"][1]
    assert not is_feasible[1]

def test_pso_objective_feasibility():
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=10.0,
        service_count=1,
        required_vcpus=1,
        required_ram_gb=1.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )
    # positions: [bandwidth, replicas]
    # Pos 0: [1000, 16] -> high cost, breaks budget 10.0
    # Pos 1: [100, 1] -> cost is 100*0.08 + 1*45 = 8+45 = 53 > 10 -> breaks budget
    pos = np.array([[1000.0, 16.0], [100.0, 1.0]])
    
    costs, is_feasible, constraints = ObjectiveEvaluator.evaluate_pso_swarm(pos, req)
    
    assert costs.shape == (2,)
    assert not is_feasible[0]
    assert not is_feasible[1]
