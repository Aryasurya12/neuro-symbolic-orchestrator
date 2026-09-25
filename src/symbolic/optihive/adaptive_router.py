"""SYM-3: Adaptive Solver Routing."""

from typing import List, Dict, Any, Tuple
from src.symbolic.models import SymbolicOptimizationRequest

class SolverRoutingFeatures:
    def __init__(self, request: SymbolicOptimizationRequest):
        # Extract deterministic features based on request
        self.resource_scale = request.required_vcpus * request.required_ram_gb
        self.service_scale = request.service_count
        # Avoid division by zero
        safe_scale = max(self.resource_scale, 1.0)
        self.budget_tightness = request.budget_max_usd / safe_scale
        self.provider_count = len(request.cloud_providers)
        self.has_latency_constraint = request.latency_max_ms > 0
        self.has_sla_constraint = request.sla_availability_pct > 0
        
        # Heuristic for constraints
        self.constraint_density = sum([
            1 if self.has_latency_constraint else 0,
            1 if self.has_sla_constraint else 0,
            1 if self.provider_count < 3 else 0,
            1 if self.budget_tightness < 5.0 else 0
        ])
        
        self.is_highly_constrained = self.constraint_density >= 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_scale": self.resource_scale,
            "service_scale": self.service_scale,
            "budget_tightness": self.budget_tightness,
            "provider_count": self.provider_count,
            "has_latency_constraint": self.has_latency_constraint,
            "has_sla_constraint": self.has_sla_constraint,
            "constraint_density": self.constraint_density,
            "is_highly_constrained": self.is_highly_constrained
        }

class SolverSuitabilityScorer:
    @staticmethod
    def score_ga(features: SolverRoutingFeatures) -> float:
        # GA is good for discrete allocation, but less so for highly constrained problems
        score = 50.0
        if features.service_scale > 5:
            score += 20.0
        if features.is_highly_constrained:
            score -= 30.0
        return max(0.0, score)
        
    @staticmethod
    def score_pso(features: SolverRoutingFeatures) -> float:
        # PSO is better for continuous tuning, which might be reflected in resource scale / less constrained
        score = 40.0
        if not features.is_highly_constrained:
            score += 20.0
        if features.budget_tightness > 10.0:
            score += 15.0
        return max(0.0, score)
        
    @staticmethod
    def score_z3(features: SolverRoutingFeatures) -> float:
        # Z3 is great for highly constrained, latency/SLA bounds
        score = 30.0
        if features.is_highly_constrained:
            score += 40.0
        if features.has_latency_constraint or features.has_sla_constraint:
            score += 20.0
        if features.service_scale > 20: # Z3 gets slow with many services
            score -= 30.0
        return max(0.0, score)

class SolverRoutingDecision:
    def __init__(
        self, 
        selected_solvers: List[str], 
        ranked_solvers: List[Tuple[str, float]],
        scores: Dict[str, float],
        features: Dict[str, Any],
        reason: str,
        confidence: str,
        routing_mode: str = "deterministic"
    ):
        self.selected_solvers = selected_solvers
        self.ranked_solvers = ranked_solvers
        self.scores = scores
        self.features = features
        self.reason = reason
        self.confidence = confidence
        self.routing_mode = routing_mode

class AdaptiveSolverRouter:
    """Adaptive Router for Optimization Engines (SYM-4 Hybrid)."""
    
    def __init__(self):
        self.learned_router = None
        try:
            from .learned_router import LearnedSolverRouter
            self.learned_router = LearnedSolverRouter()
        except Exception as e:
            print(f"Warning: LearnedSolverRouter failed to load: {e}")

    def route(self, request: SymbolicOptimizationRequest) -> SolverRoutingDecision:
        try:
            features = SolverRoutingFeatures(request)
            
            # SYM-3 Deterministic Scoring
            scores = {
                "GA": SolverSuitabilityScorer.score_ga(features),
                "PSO": SolverSuitabilityScorer.score_pso(features),
                "Z3": SolverSuitabilityScorer.score_z3(features)
            }
            
            ranked_solvers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            det_primary_solver = ranked_solvers[0][0]
            
            # Base Deterministic Decision
            det_selected_solvers = [det_primary_solver]
            if ranked_solvers[0][1] <= 70.0 or (ranked_solvers[0][1] - ranked_solvers[1][1] <= 20.0):
                if ranked_solvers[0][1] > 50.0:
                    det_selected_solvers.append(ranked_solvers[1][0])
                else:
                    det_selected_solvers = ["GA", "PSO", "Z3"]
            
            selected_solvers = list(det_selected_solvers)
            confidence = "LOW"
            reason = "Deterministic SYM-3 fallback."
            routing_mode = "deterministic"
            
            # SYM-4 Learned Routing
            if self.learned_router is not None:
                try:
                    learned_pred = self.learned_router.predict(features)
                    
                    LEARNED_HIGH_CONFIDENCE = 0.80
                    LEARNED_MEDIUM_CONFIDENCE = 0.60
                    
                    if learned_pred.confidence >= LEARNED_HIGH_CONFIDENCE:
                        routing_mode = "learned_high"
                        confidence = "HIGH"
                        reason = f"Learned router high confidence ({learned_pred.confidence:.2f})"
                        if learned_pred.predicted_solver == "GA_PSO":
                            selected_solvers = ["GA", "PSO"]
                        elif learned_pred.predicted_solver == "GA_PSO_Z3":
                            selected_solvers = ["GA", "PSO", "Z3"]
                        else:
                            selected_solvers = [learned_pred.predicted_solver]
                    
                    elif learned_pred.confidence >= LEARNED_MEDIUM_CONFIDENCE:
                        routing_mode = "learned_medium"
                        confidence = "MEDIUM"
                        reason = f"Learned router medium confidence ({learned_pred.confidence:.2f}), combined with deterministic"
                        
                        learned_solvers = []
                        if learned_pred.predicted_solver == "GA_PSO":
                            learned_solvers = ["GA", "PSO"]
                        elif learned_pred.predicted_solver == "GA_PSO_Z3":
                            learned_solvers = ["GA", "PSO", "Z3"]
                        else:
                            learned_solvers = [learned_pred.predicted_solver]
                            
                        # Combine sets
                        combined = set(learned_solvers) | set(det_selected_solvers)
                        selected_solvers = list(combined)
                except Exception as e:
                    reason = f"Learned routing failed ({e}), using deterministic."
            
            # Safety Arbitration
            # If the request requires hard verification, ensure Z3 is present.
            if features.has_latency_constraint or features.has_sla_constraint or features.is_highly_constrained:
                if "Z3" not in selected_solvers:
                    selected_solvers.append("Z3")
                    reason += " (Safety Arbitration: Added Z3 for hard constraint verification)"
                    
            return SolverRoutingDecision(
                selected_solvers=selected_solvers,
                ranked_solvers=ranked_solvers,
                scores=scores,
                features=features.to_dict(),
                reason=reason,
                confidence=confidence,
                routing_mode=routing_mode
            )
        except Exception as e:
            # Safe Fallback
            return SolverRoutingDecision(
                selected_solvers=["GA", "PSO", "Z3"],
                ranked_solvers=[],
                scores={},
                features={},
                reason=f"Routing error ({str(e)}), falling back to deterministic full race.",
                confidence="LOW",
                routing_mode="fallback"
            )
