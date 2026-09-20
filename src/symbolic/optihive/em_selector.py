"""SYM-3: Latent-Class EM Selector.

Models candidate pool as noisy observations of latent solution classes.
Provides a deterministic, probabilistic ranking of feasible candidates.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .ilp_filter import FilteredCandidate

class EMSelector:
    """Expectation-Maximization logic for candidate selection."""
    
    def __init__(self, k_classes: int = 2, max_iterations: int = 50, tolerance: float = 1e-4, seed: int = 42):
        self.k_classes = k_classes
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rng = np.random.RandomState(seed)
        
    def select(self, candidates: List[FilteredCandidate]) -> Tuple[Optional[FilteredCandidate], np.ndarray, np.ndarray]:
        """
        Selects the best feasible candidate using an EM algorithm to rank candidates.
        MUST NOT override hard feasibility. Only feasible candidates are ranked.
        Returns: (Selected Candidate, Responsibilities Matrix, Final Log-Likelihoods)
        """
        feasible_cands = [c for c in candidates if c.syntactically_feasible]
        if not feasible_cands:
            return None, np.array([]), np.array([])
            
        if len(feasible_cands) == 1:
            # Trivial case
            return feasible_cands[0], np.array([[1.0]]), np.array([0.0])
            
        X = self._extract_features(feasible_cands)
        N, D = X.shape
        
        # Initialization
        means = self.rng.rand(self.k_classes, D)
        covs = np.array([np.eye(D) for _ in range(self.k_classes)])
        weights = np.ones(self.k_classes) / self.k_classes
        
        responsibilities = np.zeros((N, self.k_classes))
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iterations):
            # E-step
            log_resp = np.zeros((N, self.k_classes))
            for k in range(self.k_classes):
                diff = X - means[k]
                cov_inv = np.linalg.pinv(covs[k] + np.eye(D) * 1e-6) # Stability
                mahala = np.sum(diff @ cov_inv * diff, axis=1)
                log_det = np.log(np.linalg.det(covs[k] + np.eye(D) * 1e-6) + 1e-10)
                log_prob = -0.5 * (mahala + log_det + D * np.log(2 * np.pi))
                log_resp[:, k] = np.log(weights[k] + 1e-10) + log_prob
                
            # Log-sum-exp trick for stability
            max_log = np.max(log_resp, axis=1, keepdims=True)
            exp_resp = np.exp(log_resp - max_log)
            sum_exp = np.sum(exp_resp, axis=1, keepdims=True)
            responsibilities = exp_resp / sum_exp
            
            log_likelihood = np.sum(max_log + np.log(sum_exp))
            
            # Convergence check
            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                break
            prev_log_likelihood = log_likelihood
            
            # M-step
            Nk = np.sum(responsibilities, axis=0) + 1e-10
            weights = Nk / N
            for k in range(self.k_classes):
                means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / Nk[k]
                diff = X - means[k]
                covs[k] = (responsibilities[:, k:k+1] * diff).T @ diff / Nk[k]
                
        # Selection: Rank by latent class probability and cost
        # We assume Class 0 corresponds to the "optimal" class (we align it heuristically)
        # We sort by cost first to ensure cost minimization remains primary within highly probable classes
        
        # We determine the "best" class as the one with lowest average cost
        class_costs = np.zeros(self.k_classes)
        for k in range(self.k_classes):
            class_costs[k] = means[k, 0] # Feature 0 is normalized cost
            
        best_class_idx = np.argmin(class_costs)
        
        # Score = (Probability of belonging to best class) - (normalized cost)
        # Higher is better
        scores = np.zeros(N)
        for i in range(N):
            prob_best = responsibilities[i, best_class_idx]
            norm_cost = X[i, 0]
            scores[i] = prob_best - (norm_cost * 0.5) 
            
        best_candidate_idx = np.argmax(scores)
        best_candidate = feasible_cands[best_candidate_idx]
        
        # Assign latent probability for provenance
        best_candidate.candidate.metadata = best_candidate.candidate.metadata or {}
        best_candidate.candidate.metadata["latent_class"] = int(best_class_idx)
        best_candidate.candidate.metadata["latent_probability"] = float(responsibilities[best_candidate_idx, best_class_idx])
        
        return best_candidate, responsibilities, np.array([log_likelihood])
        
    def _extract_features(self, candidates: List[FilteredCandidate]) -> np.ndarray:
        """Extracts normalized features for the EM model."""
        N = len(candidates)
        # Features: [Normalized Cost, Normalized Runtime]
        X = np.zeros((N, 2))
        
        costs = np.array([c.filter_objective for c in candidates])
        min_cost = np.min(costs) if len(costs) > 0 else 0
        max_cost = np.max(costs) if len(costs) > 0 else 1
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
        
        for i, c in enumerate(candidates):
            norm_cost = (c.filter_objective - min_cost) / cost_range
            
            # Mocking runtime normalized if metadata has it, else 0.5
            norm_runtime = 0.5
            if c.candidate.metadata and "runtime_ms" in c.candidate.metadata:
                norm_runtime = c.candidate.metadata["runtime_ms"] / 1000.0 # scale proxy
                
            X[i, 0] = norm_cost
            X[i, 1] = norm_runtime
            
        return X
