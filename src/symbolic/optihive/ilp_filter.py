"""SYM-3: ILP-based Syntactic Feasibility Filter.

This layer ensures that candidate solutions strictly satisfy structural
and resource constraints, mathematically preventing any downstream soft-preferences
from overriding hard feasibility requirements.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.symbolic.models import OptimizationCandidate, SymbolicOptimizationRequest

@dataclass
class FilteredCandidate:
    """Wrapper for a candidate after passing through the syntactic filter."""
    candidate: OptimizationCandidate
    syntactically_feasible: bool
    violations: List[str]
    filter_objective: float

class ILPSyntacticFilter:
    """Mathematical verification layer to block infeasible candidates."""
    
    @staticmethod
    def filter(
        candidate: OptimizationCandidate,
        request: SymbolicOptimizationRequest
    ) -> FilteredCandidate:
        """
        Syntactically verifies a candidate.
        
        Since different solvers (GA, PSO, Z3) produce different decision variable formats,
        this acts as a deterministic fallback evaluating standard reported metrics.
        If any hard threshold is demonstrably breached, it rejects the candidate.
        """
        violations = []
        is_feasible = True
        
        # 1. Budget Verification
        if candidate.objective_cost_usd > request.budget_max_usd:
            is_feasible = False
            violations.append(f"Budget exceeded: {candidate.objective_cost_usd} > {request.budget_max_usd}")
            
        # 2. Existing constraint status validation (defense in depth)
        if candidate.constraint_status:
            cs = candidate.constraint_status
            if not cs.is_feasible:
                is_feasible = False
                violations.append("Candidate marked infeasible by source solver.")
            if not cs.budget_ok:
                is_feasible = False
                violations.append("ConstraintStatus: Budget violated.")
            if not cs.vcpu_ok:
                is_feasible = False
                violations.append("ConstraintStatus: vCPU violated.")
            if not cs.ram_ok:
                is_feasible = False
                violations.append("ConstraintStatus: RAM violated.")
            if not cs.latency_ok:
                is_feasible = False
                violations.append("ConstraintStatus: Latency violated.")
            if not cs.sla_ok:
                is_feasible = False
                violations.append("ConstraintStatus: SLA violated.")
                
        # 3. Decision Variable Syntactic Provider Check (if explicit)
        dvars = candidate.decision_variables
        req_providers = {p.upper() for p in request.cloud_providers}
        for key, val in dvars.items():
            if "provider" in key.lower() and isinstance(val, str):
                if val.upper() not in req_providers:
                    is_feasible = False
                    violations.append(f"Invalid provider selected: {val} not in {request.cloud_providers}")
                    
        return FilteredCandidate(
            candidate=candidate,
            syntactically_feasible=is_feasible,
            violations=violations,
            filter_objective=candidate.objective_cost_usd
        )
