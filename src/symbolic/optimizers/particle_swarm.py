"""SYM-1: Vectorized Particle Swarm Optimization (PSO) Engine."""

import time
import numpy as np
from typing import Optional

from src.symbolic.models import (
    SymbolicOptimizationRequest,
    OptimizationResult,
    OptimizationCandidate,
    OptimizationMetrics,
    ConstraintStatus
)
from src.symbolic.interfaces import OptimizationEngine
from .objective import ObjectiveEvaluator

class ParticleSwarmOptimization(OptimizationEngine):
    def __init__(
        self,
        swarm_size: int = 30,
        iterations: int = 50,
        inertia_weight: float = 0.729,
        cognitive_coefficient: float = 1.494,
        social_coefficient: float = 1.494,
        random_seed: Optional[int] = None
    ):
        if swarm_size <= 0 or iterations <= 0:
            raise ValueError("Swarm size and iterations must be > 0.")
            
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = inertia_weight
        self.c1 = cognitive_coefficient
        self.c2 = social_coefficient
        self.rng = np.random.default_rng(random_seed)

    def solve(self, request: SymbolicOptimizationRequest, progress_callback: Optional[callable] = None) -> OptimizationResult:
        start_time = time.perf_counter()
        
        # Bounds: [bandwidth_mbps, replicas]
        # Realistic continuous bounds for this domain
        lb = np.array([100.0, 1.0])
        ub = np.array([1000.0, 16.0])

        positions = self.rng.uniform(lb, ub, (self.swarm_size, 2))
        velocities = np.zeros((self.swarm_size, 2))

        # Initial evaluation
        costs, is_feasible, constraints = ObjectiveEvaluator.evaluate_pso_swarm(positions, request)
        
        # Personal bests
        pbest_positions = positions.copy()
        pbest_costs = costs.copy()
        pbest_feasible = is_feasible.copy()

        # Global best (only among feasible if possible)
        feasible_mask = pbest_feasible
        if np.any(feasible_mask):
            feasible_indices = np.where(feasible_mask)[0]
            best_idx = feasible_indices[np.argmin(pbest_costs[feasible_indices])]
        else:
            best_idx = np.argmin(pbest_costs)
            
        gbest_position = pbest_positions[best_idx].copy()
        gbest_cost = pbest_costs[best_idx]
        gbest_feasible = pbest_feasible[best_idx]

        for iteration in range(self.iterations):
            r1 = self.rng.random((self.swarm_size, 2))
            r2 = self.rng.random((self.swarm_size, 2))

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_positions - positions)
                + self.c2 * r2 * (gbest_position - positions)
            )

            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            costs, is_feasible, constraints = ObjectiveEvaluator.evaluate_pso_swarm(positions, request)

            # Update personal bests (prefer feasible, or if both feasible/infeasible prefer lower cost)
            # Rule: 
            # If new is feasible and old is not -> update
            # If both are feasible and new cost < old cost -> update
            # If both are infeasible and new cost < old cost -> update
            update_mask = (is_feasible & ~pbest_feasible) | \
                          ((is_feasible == pbest_feasible) & (costs < pbest_costs))
            
            pbest_positions[update_mask] = positions[update_mask]
            pbest_costs[update_mask] = costs[update_mask]
            pbest_feasible[update_mask] = is_feasible[update_mask]

            # Update global best
            feasible_mask = pbest_feasible
            if np.any(feasible_mask):
                feasible_indices = np.where(feasible_mask)[0]
                current_best_idx = feasible_indices[np.argmin(pbest_costs[feasible_indices])]
                if not gbest_feasible or pbest_costs[current_best_idx] < gbest_cost:
                    gbest_cost = pbest_costs[current_best_idx]
                    gbest_position = pbest_positions[current_best_idx].copy()
                    gbest_feasible = True
            elif not gbest_feasible:
                # If everything is infeasible, track the least costly infeasible
                current_best_idx = np.argmin(pbest_costs)
                if pbest_costs[current_best_idx] < gbest_cost:
                    gbest_cost = pbest_costs[current_best_idx]
                    gbest_position = pbest_positions[current_best_idx].copy()

            if progress_callback:
                progress_callback({
                    "event": "progress",
                    "solver": "Vectorized_PSO",
                    "iteration": iteration + 1,
                    "best_cost_usd": float(gbest_cost) if gbest_feasible else None,
                    "is_feasible": bool(gbest_feasible)
                })

        runtime_ms = (time.perf_counter() - start_time) * 1000.0

        best_candidate = None
        if gbest_feasible:
            # Re-evaluate the single best to get constraint data cleanly
            best_cand_pos = gbest_position.reshape(1, 2)
            b_cost, b_feas, b_constr = ObjectiveEvaluator.evaluate_pso_swarm(best_cand_pos, request)
            
            status = ConstraintStatus(
                is_feasible=True,
                budget_ok=bool(b_constr["budget_ok"][0]),
                vcpu_ok=True, # N/A for PSO
                ram_ok=True, # N/A for PSO
                latency_ok=True,
                sla_ok=True,
                details=f"Simulated CPU: {float(b_constr['simulated_cpu'][0]):.1f}%"
            )
            
            best_candidate = OptimizationCandidate(
                decision_variables={
                    "bandwidth_mbps": float(round(gbest_position[0], 2)),
                    "replicas": float(round(gbest_position[1], 2))
                },
                objective_cost_usd=float(gbest_cost),
                is_feasible=True,
                constraint_status=status
            )

        return OptimizationResult(
            best_candidate=best_candidate,
            is_feasible=bool(gbest_feasible),
            metrics=OptimizationMetrics(runtime_ms=runtime_ms, iterations=self.iterations),
            solver_name="Vectorized_PSO"
        )
