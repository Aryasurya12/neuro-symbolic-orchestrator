"""SYM-5: Telemetry collector.

Builds a RoutingTelemetry record from OptimizationResult and routing metadata
then persists it safely. A collection failure MUST NOT fail the optimization.
"""

import time
from typing import Optional

from src.symbolic.models import OptimizationResult, SymbolicOptimizationRequest
from src.symbolic.optihive.adaptive_router import SolverRoutingFeatures
from .models import RoutingTelemetry
from .repository import TelemetryRepository

# Singleton-style default repository (can be replaced in tests)
_default_repo: Optional[TelemetryRepository] = None


def get_repository(db_path: Optional[str] = None) -> TelemetryRepository:
    global _default_repo
    if _default_repo is None or db_path is not None:
        _default_repo = TelemetryRepository(db_path=db_path) if db_path else TelemetryRepository()
    return _default_repo


def collect(
    request: SymbolicOptimizationRequest,
    result: OptimizationResult,
    routing_metadata: dict,
    total_runtime_ms: float,
    run_id: Optional[str] = None,
    scenario_tag: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Collect a telemetry record from an optimization result.

    This is failure-safe: any exception is caught and silently discarded
    so that telemetry never blocks a live optimization request.
    """
    try:
        features = SolverRoutingFeatures(request)

        is_violation = False
        obj_cost = None
        if result.best_candidate:
            cs = result.best_candidate.constraint_status
            if cs:
                is_violation = not (
                    cs.budget_ok and cs.vcpu_ok and cs.ram_ok and
                    cs.latency_ok and cs.sla_ok
                )
            obj_cost = result.best_candidate.objective_cost_usd

        record = RoutingTelemetry(
            run_id=run_id or result.metrics.__class__.__name__,  # fallback
            timestamp=time.time(),
            model_version=routing_metadata.get("routing_mode", "unknown"),
            resource_scale=features.resource_scale,
            service_scale=features.service_scale,
            budget_tightness=features.budget_tightness,
            provider_count=features.provider_count,
            has_latency_constraint=features.has_latency_constraint,
            has_sla_constraint=features.has_sla_constraint,
            constraint_density=float(features.constraint_density),
            is_highly_constrained=features.is_highly_constrained,
            selected_solvers=routing_metadata.get("selected_solvers", []),
            routing_mode=routing_metadata.get("routing_mode", "unknown"),
            learned_confidence=_parse_confidence(routing_metadata.get("confidence", "0.0")),
            fallback_used=routing_metadata.get("routing_mode", "") in ("fallback", "deterministic"),
            total_runtime_ms=total_runtime_ms,
            solver_runtime_ms=result.metrics.runtime_ms,
            is_feasible=result.is_feasible,
            hard_constraint_violation=is_violation,
            objective_cost_usd=obj_cost,
            scenario_tag=scenario_tag,
        )

        repo = get_repository(db_path=db_path)
        repo.insert(record)

    except Exception:
        # Telemetry must never break optimization
        pass


def _parse_confidence(conf) -> float:
    """Parse confidence as float. 'HIGH'/'MEDIUM'/'LOW' return proxy numeric values."""
    if isinstance(conf, float):
        return conf
    mapping = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.3}
    if isinstance(conf, str):
        return mapping.get(conf.upper(), 0.0)
    try:
        return float(conf)
    except Exception:
        return 0.0
