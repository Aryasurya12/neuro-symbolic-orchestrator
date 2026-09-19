from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SymbolicOptimizationRequest(BaseModel):
    problem_type: str
    cloud_providers: List[str]
    budget_max_usd: float
    service_count: int
    required_vcpus: int
    required_ram_gb: float
    latency_max_ms: float
    sla_availability_pct: float

class ConstraintStatus(BaseModel):
    is_feasible: bool
    budget_ok: bool
    vcpu_ok: bool
    ram_ok: bool
    latency_ok: bool
    sla_ok: bool
    details: Optional[str] = None

class OptimizationCandidate(BaseModel):
    decision_variables: Dict[str, Any]
    objective_cost_usd: float
    is_feasible: bool
    constraint_status: Optional[ConstraintStatus] = None
    metadata: Optional[Dict[str, Any]] = None

class OptimizationMetrics(BaseModel):
    runtime_ms: float
    iterations: Optional[int] = None
    evaluations: Optional[int] = None

class OptimizationResult(BaseModel):
    best_candidate: Optional[OptimizationCandidate]
    is_feasible: bool
    metrics: OptimizationMetrics
    solver_name: str
    error_message: Optional[str] = None
