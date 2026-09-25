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
        confidence: str
    ):
        self.selected_solvers = selected_solvers
        self.ranked_solvers = ranked_solvers
        self.scores = scores
        self.features = features
        self.reason = reason
        self.confidence = confidence

class AdaptiveSolverRouter:
    """Adaptive Router for Optimization Engines."""
    
    def route(self, request: SymbolicOptimizationRequest) -> SolverRoutingDecision:
        try:
            features = SolverRoutingFeatures(request)
            
            scores = {
                "GA": SolverSuitabilityScorer.score_ga(features),
                "PSO": SolverSuitabilityScorer.score_pso(features),
                "Z3": SolverSuitabilityScorer.score_z3(features)
            }
            
            ranked_solvers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            primary_solver, primary_score = ranked_solvers[0]
            secondary_solver, secondary_score = ranked_solvers[1]
            
            # Policy
            if primary_score > 70.0 and (primary_score - secondary_score > 20.0):
                confidence = "HIGH"
                selected_solvers = [primary_solver]
                reason = f"High confidence in {primary_solver} due to strong feature match."
            elif primary_score > 50.0:
                confidence = "MEDIUM"
                selected_solvers = [primary_solver, secondary_solver]
                reason = f"Medium confidence, executing {primary_solver} and {secondary_solver}."
            else:
                confidence = "LOW"
                selected_solvers = ["GA", "PSO", "Z3"]
                reason = "Low confidence across all solvers, falling back to full execution."
                
            return SolverRoutingDecision(
                selected_solvers=selected_solvers,
                ranked_solvers=ranked_solvers,
                scores=scores,
                features=features.to_dict(),
                reason=reason,
                confidence=confidence
            )
        except Exception as e:
            # Safe Fallback
            return SolverRoutingDecision(
                selected_solvers=["GA", "PSO", "Z3"],
                ranked_solvers=[],
                scores={},
                features={},
                reason=f"Routing error ({str(e)}), falling back to deterministic full race.",
                confidence="LOW"
            )
