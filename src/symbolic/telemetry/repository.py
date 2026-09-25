"""SYM-5: Telemetry repository — SQLite persistence for routing telemetry.

Telemetry is stored in a separate table within the existing cloud_finops.db
to avoid touching the pricing/provider data.
"""

import json
import sqlite3
import os
from typing import List, Optional
from .models import RoutingTelemetry

# Reuse the same db path as the existing cloud database
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "cloud_finops.db"
)

TELEMETRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS routing_telemetry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    model_version TEXT NOT NULL,

    resource_scale      REAL NOT NULL,
    service_scale       INTEGER NOT NULL,
    budget_tightness    REAL NOT NULL,
    provider_count      INTEGER NOT NULL,
    has_latency_constraint  INTEGER NOT NULL,
    has_sla_constraint      INTEGER NOT NULL,
    constraint_density      REAL NOT NULL,
    is_highly_constrained   INTEGER NOT NULL,

    selected_solvers    TEXT NOT NULL,
    routing_mode        TEXT NOT NULL,
    learned_confidence  REAL NOT NULL,
    fallback_used       INTEGER NOT NULL,

    total_runtime_ms    REAL NOT NULL,
    solver_runtime_ms   REAL NOT NULL,

    is_feasible                 INTEGER NOT NULL,
    hard_constraint_violation   INTEGER NOT NULL,
    objective_cost_usd          REAL,

    scenario_tag    TEXT
);
"""


class TelemetryRepository:
    """Persists and retrieves RoutingTelemetry records using SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._get_conn() as conn:
            conn.executescript(TELEMETRY_SCHEMA)
            conn.commit()

    def insert(self, record: RoutingTelemetry) -> None:
        sql = """
        INSERT INTO routing_telemetry (
            run_id, timestamp, model_version,
            resource_scale, service_scale, budget_tightness, provider_count,
            has_latency_constraint, has_sla_constraint,
            constraint_density, is_highly_constrained,
            selected_solvers, routing_mode, learned_confidence, fallback_used,
            total_runtime_ms, solver_runtime_ms,
            is_feasible, hard_constraint_violation, objective_cost_usd,
            scenario_tag
        ) VALUES (
            :run_id, :timestamp, :model_version,
            :resource_scale, :service_scale, :budget_tightness, :provider_count,
            :has_latency_constraint, :has_sla_constraint,
            :constraint_density, :is_highly_constrained,
            :selected_solvers, :routing_mode, :learned_confidence, :fallback_used,
            :total_runtime_ms, :solver_runtime_ms,
            :is_feasible, :hard_constraint_violation, :objective_cost_usd,
            :scenario_tag
        )
        """
        params = {
            "run_id": record.run_id,
            "timestamp": record.timestamp,
            "model_version": record.model_version,
            "resource_scale": record.resource_scale,
            "service_scale": record.service_scale,
            "budget_tightness": record.budget_tightness,
            "provider_count": record.provider_count,
            "has_latency_constraint": 1 if record.has_latency_constraint else 0,
            "has_sla_constraint": 1 if record.has_sla_constraint else 0,
            "constraint_density": record.constraint_density,
            "is_highly_constrained": 1 if record.is_highly_constrained else 0,
            "selected_solvers": json.dumps(record.selected_solvers),
            "routing_mode": record.routing_mode,
            "learned_confidence": record.learned_confidence,
            "fallback_used": 1 if record.fallback_used else 0,
            "total_runtime_ms": record.total_runtime_ms,
            "solver_runtime_ms": record.solver_runtime_ms,
            "is_feasible": 1 if record.is_feasible else 0,
            "hard_constraint_violation": 1 if record.hard_constraint_violation else 0,
            "objective_cost_usd": record.objective_cost_usd,
            "scenario_tag": record.scenario_tag,
        }
        with self._get_conn() as conn:
            conn.execute(sql, params)
            conn.commit()

    def fetch_all(self) -> List[RoutingTelemetry]:
        sql = "SELECT * FROM routing_telemetry ORDER BY timestamp ASC"
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [self._row_to_record(r) for r in rows]

    def fetch_recent(self, limit: int = 500) -> List[RoutingTelemetry]:
        sql = "SELECT * FROM routing_telemetry ORDER BY timestamp DESC LIMIT ?"
        with self._get_conn() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM routing_telemetry").fetchone()[0]

    def _row_to_record(self, row: sqlite3.Row) -> RoutingTelemetry:
        return RoutingTelemetry(
            run_id=row["run_id"],
            timestamp=row["timestamp"],
            model_version=row["model_version"],
            resource_scale=row["resource_scale"],
            service_scale=row["service_scale"],
            budget_tightness=row["budget_tightness"],
            provider_count=row["provider_count"],
            has_latency_constraint=bool(row["has_latency_constraint"]),
            has_sla_constraint=bool(row["has_sla_constraint"]),
            constraint_density=row["constraint_density"],
            is_highly_constrained=bool(row["is_highly_constrained"]),
            selected_solvers=json.loads(row["selected_solvers"]),
            routing_mode=row["routing_mode"],
            learned_confidence=row["learned_confidence"],
            fallback_used=bool(row["fallback_used"]),
            total_runtime_ms=row["total_runtime_ms"],
            solver_runtime_ms=row["solver_runtime_ms"],
            is_feasible=bool(row["is_feasible"]),
            hard_constraint_violation=bool(row["hard_constraint_violation"]),
            objective_cost_usd=row["objective_cost_usd"],
            scenario_tag=row["scenario_tag"],
        )
