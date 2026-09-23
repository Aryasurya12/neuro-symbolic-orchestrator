from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.symbolic.models import SymbolicOptimizationRequest, OptimizationResult

class OptimizationRequest(BaseModel):
    """Client request payload for optimization."""
    problem_type: str = Field(default="ILP_VM_Allocation")
    cloud_providers: List[str]
    budget_max_usd: float
    service_count: int
    required_vcpus: int
    required_ram_gb: float
    latency_max_ms: float
    sla_availability_pct: float

    def to_symbolic_request(self) -> SymbolicOptimizationRequest:
        return SymbolicOptimizationRequest(
            problem_type=self.problem_type,
            cloud_providers=self.cloud_providers,
            budget_max_usd=self.budget_max_usd,
            service_count=self.service_count,
            required_vcpus=self.required_vcpus,
            required_ram_gb=self.required_ram_gb,
            latency_max_ms=self.latency_max_ms,
            sla_availability_pct=self.sla_availability_pct
        )

class HealthResponse(BaseModel):
    status: str = "ok"

class OptimizationResponse(BaseModel):
    """The JSON response representing the final OptimizationResult."""
    solver_name: str
    is_feasible: bool
    best_candidate: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

    @classmethod
    def from_result(cls, result: OptimizationResult) -> "OptimizationResponse":
        return cls(
            solver_name=result.solver_name,
            is_feasible=result.is_feasible,
            best_candidate=result.best_candidate.model_dump() if result.best_candidate else None,
            metrics=result.metrics.model_dump(),
            error_message=result.error_message
        )

class NaturalLanguageQueryRequest(BaseModel):
    """Client request payload for natural language optimization query."""
    query: str = Field(..., description="Natural language description of cloud deployment needs.")

class NaturalLanguageQueryResponse(BaseModel):
    """The JSON response containing the generated text report."""
    report: str
