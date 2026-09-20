import asyncio
from typing import Callable, Dict, Any, Optional

from src.symbolic.models import SymbolicOptimizationRequest, OptimizationResult
from src.symbolic.optimizers.genetic_algorithm import GeneticAlgorithm
from src.symbolic.optimizers.particle_swarm import ParticleSwarmOptimization
from src.symbolic.optimizers.optimizer_race import OptimizerRace
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.optihive.solver_selection import OptiHiveSelector

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
        
        # We can run race and z3
        results = []
        
        try:
            race_result = race.execute_race(request, progress_callback=progress_callback)
            results.append(race_result)
        except Exception as e:
            if progress_callback:
                progress_callback({"event": "error", "message": f"Race error: {str(e)}"})
                
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
