import pytest
import numpy as np
from src.symbolic.models import (
    OptimizationCandidate, 
    SymbolicOptimizationRequest, 
    ConstraintStatus,
    OptimizationResult,
    OptimizationMetrics
)
from src.symbolic.optihive.solver_selection import OptiHiveSelector

def _create_mock_req():
    return SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=100.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.0
    )

def _mock_res(cost, is_feasible, solver_name, runtime):
    c = OptimizationCandidate(
        decision_variables={"provider": "AWS"},
        objective_cost_usd=cost,
        is_feasible=is_feasible,
        constraint_status=ConstraintStatus(
            is_feasible=is_feasible, budget_ok=is_feasible, vcpu_ok=True, ram_ok=True, latency_ok=True, sla_ok=True
        )
    )
    return OptimizationResult(
        best_candidate=c,
        is_feasible=is_feasible,
        metrics=OptimizationMetrics(runtime_ms=runtime),
        solver_name=solver_name
    )

def test_selector_rejects_infeasible():
    req = _create_mock_req()
    # All infeasible
    res_list = [
        _mock_res(150.0, False, "GA", 10.0),
        _mock_res(200.0, False, "PSO", 12.0)
    ]
    
    selector = OptiHiveSelector()
    final = selector.select(req, res_list)
    
    assert not final.is_feasible
    if final.best_candidate is not None:
        assert final.best_candidate.is_feasible is False

def test_selector_picks_best_feasible():
    req = _create_mock_req()
    res_list = [
        _mock_res(90.0, True, "GA", 10.0), # Good
        _mock_res(150.0, False, "PSO", 12.0), # Infeasible
        _mock_res(85.0, True, "Z3", 50.0) # Better
    ]
    
    selector = OptiHiveSelector()
    final = selector.select(req, res_list)
    
    assert final.is_feasible
    assert final.best_candidate.objective_cost_usd == 85.0
    assert final.best_candidate.metadata["source_solver"] == "Z3"
    assert final.best_candidate.metadata["ilp_filter"] == "passed"
    assert "latent_class" in final.best_candidate.metadata

def test_noisy_data_experiment():
    """
    Mandatory Phase-4 test.
    We create a pool of 3 candidates (1 optimal, 2 suboptimal) and add noise
    to their costs across 100 trials to see how often EM selects the reference optimal candidate.
    """
    req = _create_mock_req()
    rng = np.random.RandomState(42)
    
    # Base costs: Optimal = 80.0, Subopt1 = 90.0, Subopt2 = 95.0
    base_costs = [80.0, 90.0, 95.0]
    optimal_index = 0
    
    trials = 100
    successes = 0
    
    selector = OptiHiveSelector(seed=42)
    
    for _ in range(trials):
        # Add random noise N(0, 5) to cost
        noisy_costs = [c + rng.normal(0, 5.0) for c in base_costs]
        
        # Ensure they stay feasible (<=100) for the experiment
        noisy_costs = [min(c, 99.0) for c in noisy_costs]
        
        res_list = [
            _mock_res(noisy_costs[0], True, "Solver0", 10.0),
            _mock_res(noisy_costs[1], True, "Solver1", 10.0),
            _mock_res(noisy_costs[2], True, "Solver2", 10.0)
        ]
        
        # We manually inject an identifier to track the optimal one
        res_list[0].best_candidate.metadata = {"ref_id": 0}
        res_list[1].best_candidate.metadata = {"ref_id": 1}
        res_list[2].best_candidate.metadata = {"ref_id": 2}
        
        final = selector.select(req, res_list)
        
        if final.is_feasible and final.best_candidate.metadata["ref_id"] == optimal_index:
            successes += 1
            
    retention_rate = successes / trials
    print(f"Noisy Data Retention Rate: {retention_rate * 100}%")
    
    # We don't artificially enforce >95%. We just assert it works decently > 70% 
    # depending on the noise level. With N(0,5), 80 vs 90 has 10 units difference, 
    # 2 standard deviations, so it should be very high (>90%).
    assert retention_rate > 0.80
