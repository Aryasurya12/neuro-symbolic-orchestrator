import pytest
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver

def test_z3_solver_feasible():
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    assert res.solver_name == "GraphSteeredZ3"
    assert res.is_feasible
    assert res.best_candidate is not None
    assert "primary_region" in res.best_candidate.decision_variables

def test_z3_solver_unsat_budget():
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=10.0, # Impossible budget (regions cost 100+)
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    assert not res.is_feasible
    assert "UNSAT" in res.error_message

def test_z3_solver_unsat_latency():
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure", "GCP"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=1.0, # Impossible latency
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    assert not res.is_feasible
    assert "UNSAT" in res.error_message

def test_hard_vs_soft_separation_security_test():
    """
    CRITICAL MANDATORY TEST.
    Candidate A has an amazing graph score (e.g. us-east-1 and eastus have 12ms latency, penalty 12).
    Candidate B has a worse graph score (e.g. us-east-1 and eu-west-1 have 85ms latency, diverse geo -> penalty 35).
    
    We set latency_max_ms to 10.0ms. 
    Even though (us-east-1, eastus) has a much better score in reality than others, 
    it still violates the HARD constraint of <= 10.0ms.
    Therefore, NO candidate should be feasible. The soft preference MUST NOT override the hard constraint.
    """
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=10.0, # Breaks all edges
        sla_availability_pct=99.0
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    # It must be explicitly rejected
    assert not res.is_feasible
