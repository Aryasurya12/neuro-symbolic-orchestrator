"""SYM-3: OptiHive-Inspired Solver Selection Orchestrator.

Integrates multiple solver results, filters them through the ILP Syntactic Filter,
and selects the best candidate using the Latent-Class EM Selector.
"""

import time
from typing import List, Optional
from src.symbolic.models import (
    SymbolicOptimizationRequest, 
    OptimizationResult,
    OptimizationMetrics
)
from .ilp_filter import ILPSyntacticFilter
from .em_selector import EMSelector

class OptiHiveSelector:
    """Orchestrates candidate filtering and selection."""
    
    def __init__(self, k_classes: int = 2, seed: int = 42):
        self.em_selector = EMSelector(k_classes=k_classes, seed=seed)
        
    def select(
        self,
        request: SymbolicOptimizationRequest,
        results: List[OptimizationResult]
    ) -> OptimizationResult:
        """
        Consumes outputs from GA, PSO, Z3, filters infeasible candidates,
        and uses EM selection to pick the best remaining candidate.
        """
        start_time = time.perf_counter()
        
        # 1. Build Candidate Pool
        candidates = []
        for res in results:
            if res.best_candidate:
                # Store solver provenance
                res.best_candidate.metadata = res.best_candidate.metadata or {}
                res.best_candidate.metadata["source_solver"] = res.solver_name
                # Keep runtime for feature matrix
                res.best_candidate.metadata["runtime_ms"] = res.metrics.runtime_ms
                candidates.append(res.best_candidate)
                
        # 2. ILP Syntactic Filter
        filtered_candidates = []
        for cand in candidates:
            fc = ILPSyntacticFilter.filter(cand, request)
            if fc.syntactically_feasible:
                fc.candidate.metadata["ilp_filter"] = "passed"
                filtered_candidates.append(fc)
            else:
                cand.metadata["ilp_filter"] = "failed"
                cand.metadata["violations"] = fc.violations
                
        # 3. EM Latent-Class Selection
        if not filtered_candidates:
            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            best_infeasible = candidates[0] if candidates else None
            return OptimizationResult(
                best_candidate=best_infeasible,
                is_feasible=False,
                metrics=OptimizationMetrics(runtime_ms=runtime_ms),
                solver_name="OptiHive_Selector",
                error_message="No feasible candidate found."
            )
            
        best_fc, _, _ = self.em_selector.select(filtered_candidates)
        
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Add selection provenance
        best_cand = best_fc.candidate
        best_cand.metadata["selection_method"] = "opti_hive_inspired"
        
        return OptimizationResult(
            best_candidate=best_cand,
            is_feasible=True,
            metrics=OptimizationMetrics(runtime_ms=runtime_ms),
            solver_name="OptiHive_Selector"
        )
