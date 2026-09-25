import json
import os
import numpy as np

def generate_synthetic_data(num_samples: int, seed: int = 42):
    """Generates synthetic training data reflecting Phase 9/10 baseline methodology."""
    np.random.seed(seed)
    
    feature_ordering = [
        "resource_scale",
        "service_scale",
        "budget_tightness",
        "provider_count",
        "has_latency_constraint",
        "has_sla_constraint",
        "constraint_density",
        "is_highly_constrained"
    ]
    
    class_labels = ["GA_PSO", "Z3", "GA_PSO_Z3"]
    
    X = []
    y = []
    
    for _ in range(num_samples):
        # Randomize workload characteristics
        scenario_type = np.random.choice(["Unconstrained", "HighlyConstrained", "Mixed"])
        
        if scenario_type == "Unconstrained":
            resource_scale = np.random.uniform(100.0, 2048.0)
            service_scale = np.random.randint(5, 50)
            budget_tightness = np.random.uniform(5.0, 20.0)
            provider_count = np.random.randint(2, 4)
            has_latency = 0.0
            has_sla = 0.0
            target_class = 0 # GA_PSO
            
        elif scenario_type == "HighlyConstrained":
            resource_scale = np.random.uniform(4.0, 128.0)
            service_scale = np.random.randint(1, 10)
            budget_tightness = np.random.uniform(0.5, 3.0)
            provider_count = np.random.randint(1, 3)
            has_latency = 1.0
            has_sla = np.random.choice([0.0, 1.0])
            target_class = 1 # Z3
            
        else:
            resource_scale = np.random.uniform(10.0, 256.0)
            service_scale = np.random.randint(2, 15)
            budget_tightness = np.random.uniform(2.0, 6.0)
            provider_count = np.random.randint(1, 4)
            has_latency = np.random.choice([0.0, 1.0])
            has_sla = np.random.choice([0.0, 1.0])
            target_class = 2 # GA_PSO_Z3
            
        constraint_density = has_latency + has_sla + (1 if provider_count < 3 else 0) + (1 if budget_tightness < 5.0 else 0)
        is_highly_constrained = 1.0 if constraint_density >= 2 else 0.0
        
        # Override targets based on definitive logic for consistency with deterministic baseline
        if is_highly_constrained or has_latency or has_sla:
            if service_scale < 20:
                target_class = 1 # Z3
            else:
                target_class = 2 # GA_PSO_Z3
        elif constraint_density == 0:
            target_class = 0 # GA_PSO
            
        vec = [
            resource_scale,
            service_scale,
            budget_tightness,
            provider_count,
            has_latency,
            has_sla,
            constraint_density,
            is_highly_constrained
        ]
        
        X.append(vec)
        y.append(target_class)
        
    return np.array(X), np.array(y), feature_ordering, class_labels

def train_softmax_classifier(X, y, epochs=1500, lr=0.1):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # Initialize weights
    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)
    
    for _ in range(epochs):
        scores = np.dot(X, W) + b
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Gradients
        probs[np.arange(n_samples), y] -= 1
        dW = np.dot(X.T, probs) / n_samples
        db = np.sum(probs, axis=0) / n_samples
        
        # L2 Regularization
        reg = 0.01
        dW += reg * W
        
        W -= lr * dW
        b -= lr * db
        
    return W, b

def main():
    print("Generating synthetic benchmark-based training data...")
    # Train set
    X_train_raw, y_train, feature_ordering, class_labels = generate_synthetic_data(num_samples=500, seed=42)
    
    # Validation/Test set
    X_test_raw, y_test, _, _ = generate_synthetic_data(num_samples=100, seed=1337)
    
    # Normalization
    mean = np.mean(X_train_raw, axis=0)
    scale = np.std(X_train_raw, axis=0)
    safe_scale = np.where(scale == 0, 1.0, scale)
    
    X_train = (X_train_raw - mean) / safe_scale
    X_test = (X_test_raw - mean) / safe_scale
    
    print("Training model...")
    W, b = train_softmax_classifier(X_train, y_train, epochs=2000, lr=0.5)
    
    # Evaluate
    scores = np.dot(X_test, W) + b
    preds = np.argmax(scores, axis=1)
    acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    # Save checkpoint
    checkpoint_path = os.path.join(os.path.dirname(__file__), "learned_checkpoint.json")
    checkpoint = {
        "model_version": "v1.0.0",
        "feature_ordering": feature_ordering,
        "class_labels": class_labels,
        "W": W.tolist(),
        "b": b.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "metadata": {
            "training_samples": 500,
            "test_samples": 100,
            "accuracy": float(acc)
        }
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
        
    print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
