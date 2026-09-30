import pytest
from src.symbolic.interfaces import OptimizationEngine
from src.symbolic.models import SymbolicOptimizationRequest, OptimizationResult, OptimizationMetrics

class DummySolver:
    def solve(self, request: SymbolicOptimizationRequest) -> OptimizationResult:
        return OptimizationResult(
            best_candidate=None,
            is_feasible=False,
            metrics=OptimizationMetrics(runtime_ms=1.0),
            solver_name="Dummy"
        )

def test_optimization_engine_protocol():
    solver = DummySolver()
    # Verify the protocol matching
    assert isinstance(solver, OptimizationEngine)
    
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=100.0,
        service_count=1,
        required_vcpus=1,
        required_ram_gb=1.0,
        latency_max_ms=1.0,
        sla_availability_pct=99.0
    )
    res = solver.solve(req)
    assert res.solver_name == "Dummy"
    assert not res.is_feasible
