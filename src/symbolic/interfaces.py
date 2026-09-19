from typing import Protocol, runtime_checkable
from .models import SymbolicOptimizationRequest, OptimizationResult

@runtime_checkable
class OptimizationEngine(Protocol):
    """
    Common interface for all future optimization engines (GA, PSO, Z3, OptiHive).
    """
    def solve(self, request: SymbolicOptimizationRequest) -> OptimizationResult:
        ...
