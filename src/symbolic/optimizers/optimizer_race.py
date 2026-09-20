"""SYM-1: Parallel / Virtual Mathematical Race."""

import time
from typing import Optional

from src.symbolic.models import (
    SymbolicOptimizationRequest,
    OptimizationResult,
    OptimizationMetrics
)
from src.symbolic.interfaces import OptimizationEngine

class OptimizerRace:
    """
    Executes multiple optimization engines against a single request and deterministically selects the best result.
    Currently sequential, designed to be swapped with concurrent.futures if true parallelism is required later.
    """
    def __init__(self, ga_engine: OptimizationEngine, pso_engine: OptimizationEngine):
        self.ga_engine = ga_engine
        self.pso_engine = pso_engine

    def execute_race(self, request: SymbolicOptimizationRequest, progress_callback: Optional[callable] = None) -> OptimizationResult:
        """
        Executes the race. 
        Note: The problem_type generally determines which solver is most appropriate, 
        but the race architecture evaluates both to see if a crossover application yields 
        an unexpectedly viable solution.
        """
        start_time = time.perf_counter()

        # Execute sequentially for now
        ga_result = self.ga_engine.solve(request, progress_callback)
        pso_result = self.pso_engine.solve(request, progress_callback)

        # Selection logic
        best_result = self._select_best(ga_result, pso_result)

        # Update total race runtime
        total_runtime = (time.perf_counter() - start_time) * 1000.0
        best_result.metrics.runtime_ms = total_runtime

        return best_result

    def _select_best(self, res_a: OptimizationResult, res_b: OptimizationResult) -> OptimizationResult:
        """
        Deterministic Selection Policy:
        1. Feasible beats Infeasible.
        2. Lower objective cost beats higher objective cost (for feasible).
        3. Lower runtime breaks ties.
        """
        if res_a.is_feasible and not res_b.is_feasible:
            return res_a
        if res_b.is_feasible and not res_a.is_feasible:
            return res_b
        
        # Both feasible OR both infeasible
        # Compare costs
        cost_a = res_a.best_candidate.objective_cost_usd if res_a.best_candidate else float('inf')
        cost_b = res_b.best_candidate.objective_cost_usd if res_b.best_candidate else float('inf')

        if cost_a < cost_b:
            return res_a
        elif cost_b < cost_a:
            return res_b
        else:
            # Tie breaker: runtime
            if res_a.metrics.runtime_ms <= res_b.metrics.runtime_ms:
                return res_a
            return res_b
