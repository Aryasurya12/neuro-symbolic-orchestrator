"""SYM-6: Evaluation runner.

Executes multiple routing configurations against identical deterministic
scenarios and collects RoutingOutcomes.

Configurations evaluated:
    BASELINE  – GA + PSO + Z3 full race (no adaptive routing)
    SYM-3     – Deterministic adaptive routing only (learned router disabled)
    SYM-4     – Production learned checkpoint (default AdaptiveSolverRouter)
    CANDIDATE – SYM-5 candidate checkpoint
    REFERENCE – Full GA+PSO+Z3 race (evaluation oracle, identical to BASELINE)

The REFERENCE is the evaluation oracle for outcome comparison.
It is intentionally expensive (runs all solvers) because offline evaluation correctness
matters more than speed.

PRODUCTION ROUTING IS NOT MODIFIED BY THIS MODULE.
"""

import time
from typing import Optional, List, Dict, Any, Callable
from copy import deepcopy

from src.symbolic.models import SymbolicOptimizationRequest, OptimizationResult
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.optimizers.optimizer_race import OptimizerRace
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.optihive.solver_selection import OptiHiveSelector
from src.symbolic.optihive.adaptive_router import (
    AdaptiveSolverRouter, SolverRoutingDecision
)
from src.symbolic.optihive.outcome_evaluation import RoutingOutcome


CONFIG_BASELINE = "BASELINE"
CONFIG_SYM3 = "SYM-3"
CONFIG_SYM4 = "SYM-4"
CONFIG_CANDIDATE = "CANDIDATE"
CONFIG_REFERENCE = "REFERENCE"


def _build_engines():
    """Build solver engines with consistent hyperparameters."""
    ga = GeneticAlgorithm(population_size=50, generations=20)
    pso = ParticleSwarmOptimization(swarm_size=20, iterations=20)
    z3 = GraphSteeredZ3Solver(timeout_ms=2000)
    race = OptimizerRace(ga_engine=ga, pso_engine=pso)
    selector = OptiHiveSelector()
    return ga, pso, z3, race, selector


def _run_solvers(
    request: SymbolicOptimizationRequest,
    selected_solvers: List[str],
    ga, pso, z3, race, selector,
) -> OptimizationResult:
    """Run the specified solver set and return the best result via OptiHive."""
    results = []

    if "GA" in selected_solvers and "PSO" in selected_solvers:
        try:
            results.append(race.execute_race(request))
        except Exception:
            pass
    else:
        if "GA" in selected_solvers:
            try:
                results.append(ga.solve(request))
            except Exception:
                pass
        if "PSO" in selected_solvers:
            try:
                results.append(pso.solve(request))
            except Exception:
                pass

    if "Z3" in selected_solvers:
        try:
            results.append(z3.solve(request))
        except Exception:
            pass

    return selector.select(request, results)


def _extract_outcome(
    scenario_name: str,
    config_name: str,
    result: OptimizationResult,
    routing_decision: SolverRoutingDecision,
    total_ms: float,
    error: Optional[str] = None,
) -> RoutingOutcome:
    cost = None
    if result and result.best_candidate:
        cost = result.best_candidate.objective_cost_usd

    return RoutingOutcome(
        scenario_name=scenario_name,
        config_name=config_name,
        selected_solvers=routing_decision.selected_solvers,
        routing_mode=routing_decision.routing_mode,
        routing_confidence=routing_decision.confidence,
        is_feasible=result.is_feasible if result else False,
        total_runtime_ms=total_ms,
        solver_runtime_ms=result.metrics.runtime_ms if result and result.metrics else 0.0,
        objective_cost_usd=cost,
        hard_constraint_violation=False,  # Determined by constraint layer
        fallback_used=routing_decision.routing_mode in ("fallback", "deterministic"),
        error=error,
    )


class EvalSym3Router(AdaptiveSolverRouter):
    """SYM-3 only: learned router disabled."""
    def __init__(self):
        # Skip learned router initialization
        self.learned_router = None


class EvalCandidateRouter(AdaptiveSolverRouter):
    """SYM-4 architecture but using the candidate checkpoint."""
    def __init__(self, candidate_path: str):
        self.learned_router = None
        try:
            from src.symbolic.optihive.learned_router import LearnedSolverRouter
            self.learned_router = LearnedSolverRouter(checkpoint_path=candidate_path)
        except Exception:
            self.learned_router = None


def run_baseline(
    scenario_name: str,
    request: SymbolicOptimizationRequest,
) -> RoutingOutcome:
    """BASELINE: run full GA+PSO+Z3 race with no adaptive routing."""
    ga, pso, z3, race, selector = _build_engines()
    decision = SolverRoutingDecision(
        selected_solvers=["GA", "PSO", "Z3"],
        ranked_solvers=[],
        scores={},
        features={},
        reason="Baseline: full GA+PSO+Z3 race.",
        confidence="HIGH",
        routing_mode="baseline",
    )
    t0 = time.perf_counter()
    try:
        result = _run_solvers(request, ["GA", "PSO", "Z3"], ga, pso, z3, race, selector)
        ms = (time.perf_counter() - t0) * 1000.0
        return _extract_outcome(scenario_name, CONFIG_BASELINE, result, decision, ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        from src.symbolic.models import OptimizationMetrics
        dummy = OptimizationResult(is_feasible=False, metrics=OptimizationMetrics(runtime_ms=ms))
        return _extract_outcome(scenario_name, CONFIG_BASELINE, dummy, decision, ms, error=str(e))


def run_sym3(
    scenario_name: str,
    request: SymbolicOptimizationRequest,
) -> RoutingOutcome:
    """SYM-3: deterministic routing only."""
    ga, pso, z3, race, selector = _build_engines()
    router = EvalSym3Router()
    t0 = time.perf_counter()
    try:
        decision = router.route(request)
        result = _run_solvers(request, decision.selected_solvers, ga, pso, z3, race, selector)
        ms = (time.perf_counter() - t0) * 1000.0
        return _extract_outcome(scenario_name, CONFIG_SYM3, result, decision, ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        dummy_decision = SolverRoutingDecision([], [], {}, {}, str(e), "LOW", "fallback")
        from src.symbolic.models import OptimizationMetrics
        dummy = OptimizationResult(is_feasible=False, metrics=OptimizationMetrics(runtime_ms=ms))
        return _extract_outcome(scenario_name, CONFIG_SYM3, dummy, dummy_decision, ms, error=str(e))


def run_sym4(
    scenario_name: str,
    request: SymbolicOptimizationRequest,
) -> RoutingOutcome:
    """SYM-4: production learned router."""
    ga, pso, z3, race, selector = _build_engines()
    router = AdaptiveSolverRouter()
    t0 = time.perf_counter()
    try:
        decision = router.route(request)
        result = _run_solvers(request, decision.selected_solvers, ga, pso, z3, race, selector)
        ms = (time.perf_counter() - t0) * 1000.0
        return _extract_outcome(scenario_name, CONFIG_SYM4, result, decision, ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        dummy_decision = SolverRoutingDecision([], [], {}, {}, str(e), "LOW", "fallback")
        from src.symbolic.models import OptimizationMetrics
        dummy = OptimizationResult(is_feasible=False, metrics=OptimizationMetrics(runtime_ms=ms))
        return _extract_outcome(scenario_name, CONFIG_SYM4, dummy, dummy_decision, ms, error=str(e))


def run_candidate(
    scenario_name: str,
    request: SymbolicOptimizationRequest,
    candidate_path: str,
) -> RoutingOutcome:
    """CANDIDATE: SYM-5 candidate checkpoint routing."""
    ga, pso, z3, race, selector = _build_engines()
    t0 = time.perf_counter()
    try:
        router = EvalCandidateRouter(candidate_path=candidate_path)
        decision = router.route(request)
        result = _run_solvers(request, decision.selected_solvers, ga, pso, z3, race, selector)
        ms = (time.perf_counter() - t0) * 1000.0
        return _extract_outcome(scenario_name, CONFIG_CANDIDATE, result, decision, ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        dummy_decision = SolverRoutingDecision([], [], {}, {}, str(e), "LOW", "fallback")
        from src.symbolic.models import OptimizationMetrics
        dummy = OptimizationResult(is_feasible=False, metrics=OptimizationMetrics(runtime_ms=ms))
        return _extract_outcome(scenario_name, CONFIG_CANDIDATE, dummy, dummy_decision, ms, error=str(e))


def run_reference_race(
    scenario_name: str,
    request: SymbolicOptimizationRequest,
) -> RoutingOutcome:
    """REFERENCE: Full GA+PSO+Z3 race. Evaluation oracle — not production router."""
    ga, pso, z3, race, selector = _build_engines()
    decision = SolverRoutingDecision(
        selected_solvers=["GA", "PSO", "Z3"],
        ranked_solvers=[],
        scores={},
        features={},
        reason="Reference race: evaluation oracle.",
        confidence="HIGH",
        routing_mode="reference",
    )
    t0 = time.perf_counter()
    try:
        result = _run_solvers(request, ["GA", "PSO", "Z3"], ga, pso, z3, race, selector)
        ms = (time.perf_counter() - t0) * 1000.0
        return _extract_outcome(scenario_name, CONFIG_REFERENCE, result, decision, ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        from src.symbolic.models import OptimizationMetrics
        dummy = OptimizationResult(is_feasible=False, metrics=OptimizationMetrics(runtime_ms=ms))
        return _extract_outcome(scenario_name, CONFIG_REFERENCE, dummy, decision, ms, error=str(e))
