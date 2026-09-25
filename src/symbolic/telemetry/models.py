"""SYM-5: Telemetry data models for routing feedback."""

from pydantic import BaseModel, Field
from typing import List, Optional
import time
import uuid


class RoutingTelemetry(BaseModel):
    """A single recorded routing telemetry event.
    
    Captures everything needed to reconstruct the routing decision and its outcome.
    Does NOT store raw natural-language prompts — only structured features and metrics.
    """

    # Identity
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    model_version: str = "unknown"

    # Feature vector (mirrors SolverRoutingFeatures)
    resource_scale: float = 0.0
    service_scale: int = 0
    budget_tightness: float = 0.0
    provider_count: int = 0
    has_latency_constraint: bool = False
    has_sla_constraint: bool = False
    constraint_density: float = 0.0
    is_highly_constrained: bool = False

    # Routing decision
    selected_solvers: List[str] = Field(default_factory=list)
    routing_mode: str = "unknown"          # learned_high / learned_medium / deterministic / fallback
    learned_confidence: float = 0.0        # Raw probability from learned model (0 if not used)
    fallback_used: bool = False

    # Execution outcomes
    total_runtime_ms: float = 0.0
    solver_runtime_ms: float = 0.0

    # Feasibility (set by hard constraint layer — NOT by telemetry)
    is_feasible: bool = False
    hard_constraint_violation: bool = False
    objective_cost_usd: Optional[float] = None

    # Optional scenario tag (for benchmark identification)
    scenario_tag: Optional[str] = None
