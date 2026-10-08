from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class FeasibilityStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    SATISFIABLE = "SATISFIABLE"
    UNKNOWN = "UNKNOWN"

class SymbolicOptimizationRequest(BaseModel):
    problem_type: str
    cloud_providers: List[str]
    budget_max_usd: float
    service_count: int = 1
    required_vcpus: int = 1
    required_ram_gb: float = 1.0
    latency_max_ms: float = 100.0
    sla_availability_pct: float = 99.9
    target_bandwidth_mbps: Optional[float] = None
    min_bandwidth_mbps: Optional[float] = None
    max_bandwidth_mbps: Optional[float] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    target_replicas: Optional[int] = None
    target_cpu_pct: Optional[float] = None
    max_cpu_pct: Optional[float] = None
    primary_region: Optional[str] = None
    secondary_region: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

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
    status: Optional[FeasibilityStatus] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
