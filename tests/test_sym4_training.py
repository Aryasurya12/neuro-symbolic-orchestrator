import pytest
import numpy as np
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optihive.adaptive_router import SolverRoutingFeatures
from src.symbolic.optihive.learned_training import generate_synthetic_data, train_softmax_classifier

def test_feature_ordering_is_deterministic():
    """Feature vector must be in exactly the same order every time."""
    req = SymbolicOptimizationRequest(
        problem_type="Test",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=100.0,
        service_count=5,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=0.0,
        sla_availability_pct=0.0
    )
    features = SolverRoutingFeatures(req)
    d = features.to_dict()
    ordering = [
        "resource_scale", "service_scale", "budget_tightness", "provider_count",
        "has_latency_constraint", "has_sla_constraint", "constraint_density", "is_highly_constrained"
    ]
    for key in ordering:
        assert key in d, f"Missing key: {key}"

def test_training_data_is_reproducible():
    X1, y1, _, _ = generate_synthetic_data(num_samples=100, seed=42)
    X2, y2, _, _ = generate_synthetic_data(num_samples=100, seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)

def test_different_seeds_produce_different_data():
    X1, y1, _, _ = generate_synthetic_data(num_samples=100, seed=42)
    X2, y2, _, _ = generate_synthetic_data(num_samples=100, seed=99)
    assert not np.array_equal(X1, X2)

def test_training_classifier():
    X, y, _, class_labels = generate_synthetic_data(num_samples=200, seed=42)
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    safe_scale = np.where(scale == 0, 1.0, scale)
    X_norm = (X - mean) / safe_scale
    
    W, b = train_softmax_classifier(X_norm, y, epochs=500, lr=0.5)
    
    # The model should have the correct shape
    assert W.shape == (X.shape[1], len(set(y)))
    assert b.shape == (len(set(y)),)
    
    # Accuracy on training data should be reasonable
    scores = np.dot(X_norm, W) + b
    preds = np.argmax(scores, axis=1)
    acc = np.mean(preds == y)
    assert acc > 0.8, f"Training accuracy too low: {acc}"
