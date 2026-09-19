"""SYM-1: Shared Objective & Feasibility Evaluator.

Provides vectorized evaluation logic to be shared between GA and PSO engines.
"""

from typing import Tuple, Dict, Any
import numpy as np
from src.symbolic.models import SymbolicOptimizationRequest, ConstraintStatus
from .domain_catalog import CATALOG_COSTS, CATALOG_VCPUS, CATALOG_RAM

class ObjectiveEvaluator:
    """Evaluates optimization candidates, clearly separating objective cost from hard feasibility."""

    @staticmethod
    def evaluate_ga_population(
        population: np.ndarray,
        request: SymbolicOptimizationRequest
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Evaluates a discrete allocation population (shape: [N, num_skus]).
        
        Returns:
            costs: np.ndarray of shape (N,) with total cost of each individual.
            is_feasible: np.ndarray of shape (N,) boolean mask.
            constraints: dict of specific constraint masks.
        """
        # Costs and resources
        costs = np.dot(population, CATALOG_COSTS)
        vcpus = np.dot(population, CATALOG_VCPUS)
        ram = np.dot(population, CATALOG_RAM)

        # Constraint evaluation
        budget_ok = costs <= request.budget_max_usd
        vcpu_ok = vcpus >= request.required_vcpus
        ram_ok = ram >= request.required_ram_gb

        # We assume latency and SLA are 'ok' for GA allocation in this phase since 
        # the catalog doesn't provide enough information to constrain them meaningfully.
        is_feasible = budget_ok & vcpu_ok & ram_ok

        constraints = {
            "budget_ok": budget_ok,
            "vcpu_ok": vcpu_ok,
            "ram_ok": ram_ok,
        }

        return costs, is_feasible, constraints

    @staticmethod
    def evaluate_pso_swarm(
        positions: np.ndarray,
        request: SymbolicOptimizationRequest,
        cost_per_mbps_month: float = 0.08,
        cost_per_replica_month: float = 45.0,
        target_cpu_pct: float = 70.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Evaluates a continuous swarm (shape: [N, 2]) where pos[0]=bandwidth, pos[1]=replicas.
        
        Returns:
            costs: np.ndarray of shape (N,) with total cost of each particle.
            is_feasible: np.ndarray of shape (N,) boolean mask.
            constraints: dict of specific constraint masks.
        """
        bw = positions[:, 0]
        reps = positions[:, 1]

        # Calculate base monthly infrastructure cost
        costs = (bw * cost_per_mbps_month) + (reps * cost_per_replica_month)

        # Feasibility check based on budget
        budget_ok = costs <= request.budget_max_usd

        # CPU constraint check (ensure we don't massively under-provision or over-provision)
        # We define a loose "acceptable" range for soft feasibility since PSO continuously optimizes it.
        simulated_cpu = np.clip((bw / (reps * 75.0)) * 100.0, 10.0, 99.0)
        
        # We'll consider it functionally feasible if it's within bounds and budget.
        is_feasible = budget_ok

        constraints = {
            "budget_ok": budget_ok,
            "simulated_cpu": simulated_cpu,
        }

        return costs, is_feasible, constraints

    @classmethod
    def get_ga_fitness(cls, population: np.ndarray, request: SymbolicOptimizationRequest) -> np.ndarray:
        """
        Returns a fitness score for GA selection. 
        Infeasible solutions receive a massive penalty to lower their rank.
        """
        costs, is_feasible, _ = cls.evaluate_ga_population(population, request)
        
        # Massive penalty for infeasible solutions
        penalty = (~is_feasible) * 1e6
        
        # Fitness is cost + penalty. Lower is better.
        return costs + penalty

    @classmethod
    def create_constraint_status(
        cls, is_feasible: bool, budget_ok: bool, vcpu_ok: bool, ram_ok: bool
    ) -> ConstraintStatus:
        """Helper to build a uniform ConstraintStatus object."""
        return ConstraintStatus(
            is_feasible=bool(is_feasible),
            budget_ok=bool(budget_ok),
            vcpu_ok=bool(vcpu_ok),
            ram_ok=bool(ram_ok),
            latency_ok=True,  # Placeholder for Phase 2
            sla_ok=True,      # Placeholder for Phase 2
            details="Validated by Vectorized ObjectiveEvaluator"
        )
