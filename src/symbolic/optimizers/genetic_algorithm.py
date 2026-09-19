"""SYM-1: Vectorized Genetic Algorithm (GA) Optimization Engine."""

import time
import numpy as np
from typing import Optional

from src.symbolic.models import (
    SymbolicOptimizationRequest,
    OptimizationResult,
    OptimizationCandidate,
    OptimizationMetrics
)
from src.symbolic.interfaces import OptimizationEngine
from .domain_catalog import VM_CATALOG, get_provider_mask
from .objective import ObjectiveEvaluator

class GeneticAlgorithm(OptimizationEngine):
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_count: int = 2,
        random_seed: Optional[int] = None
    ):
        if population_size <= 0 or generations <= 0:
            raise ValueError("Population size and generations must be > 0.")
        if not (0.0 <= mutation_rate <= 1.0) or not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("Rates must be between 0.0 and 1.0.")
            
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = min(elite_count, population_size)
        self.rng = np.random.default_rng(random_seed)
        self.num_genes = len(VM_CATALOG)

    def _initialize_population(self, max_units: int, mask: np.ndarray) -> np.ndarray:
        """Initialize random population, masking out invalid providers."""
        pop = self.rng.integers(0, max_units + 1, size=(self.population_size, self.num_genes))
        # Zero out SKUs that aren't from the requested cloud providers
        pop[:, ~mask] = 0
        return pop

    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray, k: int = 3) -> np.ndarray:
        """Vectorized tournament selection."""
        # Randomly select k individuals for each tournament
        tournaments = self.rng.integers(0, self.population_size, size=(self.population_size, k))
        # Find the index of the best fitness in each tournament
        best_indices = np.argmin(fitness[tournaments], axis=1)
        # Gather the winning indices
        winner_indices = tournaments[np.arange(self.population_size), best_indices]
        return population[winner_indices]

    def _uniform_crossover(self, parents: np.ndarray) -> np.ndarray:
        """Vectorized uniform crossover."""
        children = parents.copy()
        # Create pairs
        parent1 = children[::2]
        parent2 = children[1::2]
        
        # Crossover mask
        do_crossover = self.rng.random(size=parent1.shape[0]) < self.crossover_rate
        swap_mask = self.rng.choice([True, False], size=parent1.shape)
        
        # Apply crossover only to selected pairs
        swap_mask = swap_mask & do_crossover[:, np.newaxis]
        
        # Swap genes
        parent1_new = np.where(swap_mask, parent2, parent1)
        parent2_new = np.where(swap_mask, parent1, parent2)
        
        children[::2] = parent1_new
        children[1::2] = parent2_new
        return children

    def _mutation(self, population: np.ndarray, max_units: int, provider_mask: np.ndarray) -> np.ndarray:
        """Vectorized uniform mutation."""
        mutate_mask = self.rng.random(size=population.shape) < self.mutation_rate
        # We only mutate valid providers
        mutate_mask[:, ~provider_mask] = False
        
        random_genes = self.rng.integers(0, max_units + 1, size=population.shape)
        return np.where(mutate_mask, random_genes, population)

    def solve(self, request: SymbolicOptimizationRequest) -> OptimizationResult:
        start_time = time.perf_counter()
        
        provider_mask = get_provider_mask(request.cloud_providers)
        if not np.any(provider_mask):
            return OptimizationResult(
                best_candidate=None,
                is_feasible=False,
                metrics=OptimizationMetrics(runtime_ms=(time.perf_counter() - start_time) * 1000.0),
                solver_name="Vectorized_GA",
                error_message="No VM SKUs available for requested cloud providers."
            )

        # Estimate a reasonable upper bound for instances to search over
        min_vcpu = np.min([sku.vcpus for sku in VM_CATALOG])
        max_units = min(8, max(2, int(np.ceil(request.required_vcpus / min_vcpu))))

        population = self._initialize_population(max_units, provider_mask)
        
        best_candidate = None
        best_cost = float('inf')
        best_feasible = False

        for _ in range(self.generations):
            costs, is_feasible, constraints = ObjectiveEvaluator.evaluate_ga_population(population, request)
            fitness = costs + (~is_feasible) * 1e6

            # Track generation best
            gen_best_idx = np.argmin(fitness)
            if is_feasible[gen_best_idx] and costs[gen_best_idx] < best_cost:
                best_cost = costs[gen_best_idx]
                best_feasible = True
                best_chrom = population[gen_best_idx].copy()
                
                # Build candidate
                decisions = {
                    VM_CATALOG[i].name: int(best_chrom[i]) 
                    for i in range(self.num_genes) if best_chrom[i] > 0
                }
                status = ObjectiveEvaluator.create_constraint_status(
                    is_feasible=True,
                    budget_ok=bool(constraints["budget_ok"][gen_best_idx]),
                    vcpu_ok=bool(constraints["vcpu_ok"][gen_best_idx]),
                    ram_ok=bool(constraints["ram_ok"][gen_best_idx])
                )
                best_candidate = OptimizationCandidate(
                    decision_variables=decisions,
                    objective_cost_usd=float(best_cost),
                    is_feasible=True,
                    constraint_status=status
                )

            # Elitism
            elite_indices = np.argsort(fitness)[:self.elite_count]
            elites = population[elite_indices].copy()

            # Next generation
            parents = self._tournament_selection(population, fitness)
            children = self._uniform_crossover(parents)
            population = self._mutation(children, max_units, provider_mask)
            
            # Restore elites
            population[:self.elite_count] = elites

        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        return OptimizationResult(
            best_candidate=best_candidate,
            is_feasible=best_feasible,
            metrics=OptimizationMetrics(runtime_ms=runtime_ms, iterations=self.generations),
            solver_name="Vectorized_GA"
        )
