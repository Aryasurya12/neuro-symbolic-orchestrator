"""Tests for Z3 Learned Graph Steering."""

import pytest
import z3
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver

def get_base_request() -> SymbolicOptimizationRequest:
    return SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS", "Azure", "GCP"],
        budget_max_usd=1500.0,
        service_count=1,
        required_vcpus=16,
        required_ram_gb=32.0,
        latency_max_ms=100.0,
        sla_availability_pct=95.0
    )

def test_hard_budget_protection():
    req = get_base_request()
    req.budget_max_usd = 10.0 # Way too low
    engine = GraphSteeredZ3Solver(steering_mode="learned")
    res = engine.solve(req)
    assert not res.is_feasible
    assert res.best_candidate is None

def test_hard_latency_protection():
    req = get_base_request()
    req.latency_max_ms = 1.0 # Unattainable
    engine = GraphSteeredZ3Solver(steering_mode="learned")
    res = engine.solve(req)
    assert not res.is_feasible

def test_hard_sla_protection():
    req = get_base_request()
    req.sla_availability_pct = 100.0 # Truly unattainable with 99.95% nodes
    engine = GraphSteeredZ3Solver(steering_mode="learned")
    res = engine.solve(req)
    assert not res.is_feasible

def test_provider_constraint_protection():
    req = get_base_request()
    req.cloud_providers = ["UnknownProvider"]
    engine = GraphSteeredZ3Solver(steering_mode="learned")
    res = engine.solve(req)
    assert not res.is_feasible

def test_fallback_behavior():
    engine = GraphSteeredZ3Solver(steering_mode="learned")
    # Manually break model to simulate failure
    engine.steering._learned_embeddings = None
    req = get_base_request()
    res = engine.solve(req)
    assert res.is_feasible
    # Ensures it fell back without crashing

def test_infeasible_returns_unsat():
    req = get_base_request()
    req.budget_max_usd = 0.1
    engine = GraphSteeredZ3Solver(steering_mode="learned")
    res = engine.solve(req)
    assert res.is_feasible is False
