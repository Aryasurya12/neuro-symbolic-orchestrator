import pytest
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optihive.learned_router import LearnedSolverRouter
from src.symbolic.optihive.adaptive_router import AdaptiveSolverRouter

def test_missing_checkpoint_fallback(tmp_path):
    # Pass a path that doesn't exist
    with pytest.raises(FileNotFoundError):
        LearnedSolverRouter(checkpoint_path=str(tmp_path / "does_not_exist.json"))

def test_adaptive_router_safe_fallback_on_exception(monkeypatch):
    router = AdaptiveSolverRouter()
    
    # Force an exception inside the learned router
    def mock_predict(*args, **kwargs):
        raise ValueError("Injected failure")
        
    if router.learned_router:
        monkeypatch.setattr(router.learned_router, "predict", mock_predict)
        
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=100.0,
        service_count=1,
        required_vcpus=2,
        required_ram_gb=4.0,
        latency_max_ms=0.0,
        sla_availability_pct=0.0
    )
    
    decision = router.route(req)
    # Even if learned routing fails, it should seamlessly fall back to deterministic
    assert decision.reason.startswith("Learned routing failed") or decision.routing_mode == "deterministic"
    assert len(decision.selected_solvers) > 0

def test_safety_arbitration_hard_constraints():
    router = AdaptiveSolverRouter()
    
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=100.0,
        service_count=1,
        required_vcpus=2,
        required_ram_gb=4.0,
        latency_max_ms=10.0,  # Latency constraint is present
        sla_availability_pct=0.0
    )
    
    decision = router.route(req)
    # Core guarantee: Z3 must always be present for latency-constrained requests.
    # Either the learned model already included it (learned_high mode), or
    # safety arbitration appended it. Either outcome is correct.
    assert "Z3" in decision.selected_solvers
