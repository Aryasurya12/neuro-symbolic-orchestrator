import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from .adaptive_router import SolverRoutingFeatures

class LearnedRoutingPrediction:
    def __init__(
        self,
        predicted_solver: str,
        probabilities: Dict[str, float],
        confidence: float,
        model_version: str
    ):
        self.predicted_solver = predicted_solver
        self.probabilities = probabilities
        self.confidence = confidence
        self.model_version = model_version

class LearnedSolverRouter:
    """Learned routing layer for Adaptive Solver Routing."""
    
    def __init__(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "learned_checkpoint.json")
            
        self.checkpoint_path = checkpoint_path
        self._load_checkpoint()
        
    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        with open(self.checkpoint_path, 'r') as f:
            data = json.load(f)
            
        self.model_version = data.get("model_version", "unknown")
        self.feature_ordering = data["feature_ordering"]
        self.class_labels = data["class_labels"]
        self.W = np.array(data["W"])
        self.b = np.array(data["b"])
        self.mean = np.array(data["mean"])
        self.scale = np.array(data["scale"])
        
    def _extract_vector(self, features: SolverRoutingFeatures) -> np.ndarray:
        feature_dict = features.to_dict()
        vec = []
        for feat in self.feature_ordering:
            val = feature_dict.get(feat, 0)
            if isinstance(val, bool):
                vec.append(1.0 if val else 0.0)
            else:
                vec.append(float(val))
        return np.array(vec)
        
    def predict(self, features: SolverRoutingFeatures) -> LearnedRoutingPrediction:
        try:
            # 1. Feature Extraction
            X_raw = self._extract_vector(features)
            
            # 2. Normalization
            # Handle potential division by zero in scale if a feature is constant
            safe_scale = np.where(self.scale == 0, 1.0, self.scale)
            X_norm = (X_raw - self.mean) / safe_scale
            
            # 3. Softmax Classifier (Logistic Regression) Inference
            # score = X * W + b
            scores = np.dot(X_norm, self.W) + self.b
            
            # stability shift
            scores -= np.max(scores)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)
            
            # 4. Result interpretation
            pred_idx = int(np.argmax(probs))
            predicted_class = self.class_labels[pred_idx]
            confidence = float(probs[pred_idx])
            
            prob_dict = {self.class_labels[i]: float(probs[i]) for i in range(len(self.class_labels))}
            
            return LearnedRoutingPrediction(
                predicted_solver=predicted_class,
                probabilities=prob_dict,
                confidence=confidence,
                model_version=self.model_version
            )
            
        except Exception as e:
            raise RuntimeError(f"Learned inference failed: {str(e)}")
