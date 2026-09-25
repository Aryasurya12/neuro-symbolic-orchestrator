import pytest
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optihive.adaptive_router import AdaptiveSolverRouter, SolverRoutingFeatures

def test_routing_features():
    req = SymbolicOptimizationRequest(
        problem_type="TEST",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=100.0,
        service_count=5,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.9
    )
    features = SolverRoutingFeatures(req)
    
    # resource_scale = 4 * 16 = 64
    assert features.resource_scale == 64.0
    # budget_tightness = 100 / 64 = 1.5625
    assert abs(features.budget_tightness - 1.5625) < 0.01
    assert features.provider_count == 2
    assert features.has_latency_constraint is True
    assert features.has_sla_constraint is True
    assert features.is_highly_constrained is True

def test_adaptive_router_highly_constrained():
    req = SymbolicOptimizationRequest(
        problem_type="TEST",
        cloud_providers=["AWS"],
        budget_max_usd=100.0,
        service_count=2,
        required_vcpus=2,
        required_ram_gb=4.0,
        latency_max_ms=10.0,
        sla_availability_pct=99.99
    )
    router = AdaptiveSolverRouter()
    decision = router.route(req)
    
    # Highly constrained -> Z3 should win
    assert "Z3" in decision.selected_solvers
    assert decision.ranked_solvers[0][0] == "Z3"

def test_adaptive_router_unconstrained_scale():
    req = SymbolicOptimizationRequest(
        problem_type="TEST",
        cloud_providers=["AWS", "Azure", "GCP"],
        budget_max_usd=10000.0,
        service_count=20,
        required_vcpus=100,
        required_ram_gb=400.0,
        latency_max_ms=0.0,
        sla_availability_pct=0.0
    )
    router = AdaptiveSolverRouter()
    decision = router.route(req)
    
    # Unconstrained, discrete -> GA or PSO should win. Z3 gets penalized for high service scale.
    assert "GA" in decision.selected_solvers or "PSO" in decision.selected_solvers
    assert decision.ranked_solvers[-1][0] == "Z3" # Z3 should be last

def test_adaptive_router_fallback():
    # A malformed request should fall back gracefully
    # Pydantic might catch it, but if we somehow bypass or error during routing:
    class BadReq:
        pass
    
    router = AdaptiveSolverRouter()
    decision = router.route(BadReq())
    
    assert decision.confidence == "LOW"
    assert "GA" in decision.selected_solvers
    assert "PSO" in decision.selected_solvers
    assert "Z3" in decision.selected_solvers
