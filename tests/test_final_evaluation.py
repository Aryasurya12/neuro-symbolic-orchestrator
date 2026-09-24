import pytest
from benchmarks.final_evaluation import SCENARIOS

def test_scenario_definitions_complete():
    """Verify all scenarios have required deterministic fields."""
    assert len(SCENARIOS) == 13
    
    categories = [sc["category"] for sc in SCENARIOS]
    assert "Feasible" in categories
    assert "Budget Stress" in categories
    assert "Infeasible" in categories
    assert "Multi-Provider" in categories
    assert "Latency/SLA" in categories
    assert "Boundary" in categories
    
    for sc in SCENARIOS:
        assert "scenario_id" in sc
        assert "query" in sc
        assert "expected_feasible" in sc

def test_baseline_deterministic():
    """Ensure baseline scenarios use exact text so the semantic layer produces expected contracts."""
    feasible_small = next(sc for sc in SCENARIOS if sc["scenario_id"] == "A1_Small")
    assert feasible_small["query"] == "Deploy a microservice needing 2 vCPUs and 4GB RAM on AWS for under $200 a month."
    
    boundary_budget = next(sc for sc in SCENARIOS if sc["scenario_id"] == "F1_Boundary_Budget")
    assert not boundary_budget["expected_feasible"]
