import pytest
import os
import json
import numpy as np
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optihive.adaptive_router import SolverRoutingFeatures, AdaptiveSolverRouter
from src.symbolic.optihive.learned_router import LearnedSolverRouter

@pytest.fixture
def mock_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "learned_checkpoint.json"
    feature_ordering = [
        "resource_scale", "service_scale", "budget_tightness", "provider_count",
        "has_latency_constraint", "has_sla_constraint", "constraint_density", "is_highly_constrained"
    ]
    class_labels = ["GA_PSO", "Z3", "GA_PSO_Z3"]
    # Mock parameters
    W = np.zeros((8, 3)).tolist()
    b = [0.1, 0.2, 0.7] # GA_PSO_Z3 is highest natively
    mean = [0]*8
    scale = [1]*8
    
    data = {
        "model_version": "v1.0.0",
        "feature_ordering": feature_ordering,
        "class_labels": class_labels,
        "W": W,
        "b": b,
        "mean": mean,
        "scale": scale
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f)
        
    return str(checkpoint_path)

def test_learned_router_inference(mock_checkpoint):
    router = LearnedSolverRouter(checkpoint_path=mock_checkpoint)
    
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
    features = SolverRoutingFeatures(req)
    
    pred = router.predict(features)
    assert pred.predicted_solver == "GA_PSO_Z3"
    assert "GA_PSO" in pred.probabilities
    assert pred.confidence > 0.0

def test_hybrid_router_integration(mock_checkpoint):
    # Temporarily replace the default checkpoint path in LearnedSolverRouter 
    # to use our mock, or we can just test the real one if it exists.
    # We will test the real one since the prompt requires it to be present.
    router = AdaptiveSolverRouter()
    
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS"],
        budget_max_usd=100.0,
        service_count=1,
        required_vcpus=2,
        required_ram_gb=4.0,
        latency_max_ms=10.0, # Highly constrained -> forces Z3 safety
        sla_availability_pct=99.9
    )
    
    decision = router.route(req)
    # Z3 must be included due to safety arbitration
    assert "Z3" in decision.selected_solvers
    assert decision.confidence in ["LOW", "MEDIUM", "HIGH"]
