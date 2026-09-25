import asyncio
from typing import Callable, Dict, Any, Optional

from src.symbolic.models import SymbolicOptimizationRequest, OptimizationResult
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.optimizers.optimizer_race import OptimizerRace
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.optihive.solver_selection import OptiHiveSelector
from src.symbolic.optihive.adaptive_router import AdaptiveSolverRouter

class OptimizationService:
    """Orchestrates the existing symbolic optimization engines."""
    
    def __init__(self):
        # We instantiate the engines once or per-request. Per-request is safer for stateless isolation.
        pass

    def run_pipeline_sync(
        self, 
        request: SymbolicOptimizationRequest, 
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> OptimizationResult:
        """
        Runs the full Phase 1-4 pipeline synchronously.
        This must be called via asyncio.to_thread in FastAPI.
        """
        ga_engine = GeneticAlgorithm(population_size=50, generations=20)
        pso_engine = ParticleSwarmOptimization(swarm_size=20, iterations=20)
        z3_engine = GraphSteeredZ3Solver(timeout_ms=2000)
        
        race = OptimizerRace(ga_engine=ga_engine, pso_engine=pso_engine)
        
        # SYM-3: Adaptive Routing
        if progress_callback:
            progress_callback({"event": "router_started", "message": "Evaluating solver suitability"})
        
        router = AdaptiveSolverRouter()
        routing_decision = router.route(request)
        
        if progress_callback:
            progress_callback({
                "event": "routing_decision", 
                "selected_solvers": routing_decision.selected_solvers,
                "confidence": routing_decision.confidence,
                "reason": routing_decision.reason
            })
            
        results = []
        
        # Execute selectively based on routing decision
        if "GA" in routing_decision.selected_solvers and "PSO" in routing_decision.selected_solvers:
            # If both are selected, we can still use the race architecture for efficiency
            try:
                race_result = race.execute_race(request, progress_callback=progress_callback)
                results.append(race_result)
            except Exception as e:
                if progress_callback:
                    progress_callback({"event": "error", "message": f"Race error: {str(e)}"})
        else:
            if "GA" in routing_decision.selected_solvers:
                try:
                    ga_result = ga_engine.solve(request, progress_callback=progress_callback)
                    results.append(ga_result)
                except Exception as e:
                    if progress_callback:
                        progress_callback({"event": "error", "message": f"GA error: {str(e)}"})
            if "PSO" in routing_decision.selected_solvers:
                try:
                    pso_result = pso_engine.solve(request, progress_callback=progress_callback)
                    results.append(pso_result)
                except Exception as e:
                    if progress_callback:
                        progress_callback({"event": "error", "message": f"PSO error: {str(e)}"})
                        
        if "Z3" in routing_decision.selected_solvers:
            try:
                z3_result = z3_engine.solve(request, progress_callback=progress_callback)
                results.append(z3_result)
            except Exception as e:
                if progress_callback:
                    progress_callback({"event": "error", "message": f"Z3 error: {str(e)}"})
                
        # Phase 4 OptiHive selection
        if progress_callback:
            progress_callback({"event": "solver_started", "solver": "OptiHiveSelector"})
            
        selector = OptiHiveSelector()
        final_result = selector.select(request, results)
        
        # Attach routing metadata
        if final_result.best_candidate:
            final_result.best_candidate.metadata = final_result.best_candidate.metadata or {}
            final_result.best_candidate.metadata["routing"] = {
                "selected_solvers": routing_decision.selected_solvers,
                "ranked_solvers": routing_decision.ranked_solvers,
                "scores": routing_decision.scores,
                "features": routing_decision.features,
                "reason": routing_decision.reason,
                "confidence": routing_decision.confidence
            }
        
        if progress_callback:
            progress_callback({"event": "solver_completed", "solver": "OptiHiveSelector"})
            
        return final_result

    async def run_pipeline_async(
        self, 
        request: SymbolicOptimizationRequest,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> OptimizationResult:
        """Runs the pipeline in a separate thread to prevent event loop blocking."""
        return await asyncio.to_thread(self.run_pipeline_sync, request, progress_callback)
